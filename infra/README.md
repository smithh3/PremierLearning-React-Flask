# Premierlearning on AWS

Phase 0 is a **single Fargate task** that can suggest or apply transfers (and optionally train first). Phase 1 adds a weekly schedule. Phase 2 splits learn and apply behind Postgres + SQS.

Stay in **eu-west-2 (London)**. Fargate has no public IP; outbound FPL traffic goes through a small always-on NAT EC2 so FPL sees that address (Fargate IPs were 403).

## 1. Account hygiene (do this in the console first)

Use the personal AWS account you just created. Sign in as root only for these steps, then stop using root.

1. Region selector (top right) → **Europe (London) eu-west-2**.
2. Search **IAM** → Dashboard → **Add MFA** for the root user (authenticator app or phone).
3. Search **Billing** → Billing preferences → enable **Receive Billing Alerts**.
4. Search **IAM** → Users → **Create user** (e.g. `hal-admin`).
   - Enable console password.
   - Attach policy **AdministratorAccess** (fine for a one-person account).
   - Security credentials → **Create access key** → Command Line Interface.
   - Store the key on your machine only. Never commit it.
5. Sign out of root. Sign in as `hal-admin`.
6. Install [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), then:

```bash
aws configure
# AWS Access Key ID: (paste)
# AWS Secret Access Key: (paste)
# Default region name: eu-west-2
# Default output format: json

aws sts get-caller-identity
```

You should see your 12-digit account id and `hal-admin`. If this command fails, stop; ECS will not work until it succeeds.

Optional: put your email in Terraform `billing_email` so a **$10 estimated-charges** alarm is created (billing metrics live in `us-east-1`; confirm the SNS subscription in your inbox).

## 2. Local prove (no AWS)

Needs Docker Desktop. This is the GW1-critical test: FPL login and squadtools against a real database, on your machine.

```bash
cp .env.example .env
# Edit .env: FPL_EMAIL, FPL_PASSWORD. Leave APPLY_TRANSFERS=false.

# If you already trained locally, copy the sqlite file into database/fantasy.db
# (learning writes ../database/fantasy.db from the api/ working directory).

docker compose build
docker compose run --rm pipeline
```

Default is **suggest only**. You should see a login, then printed transfers, and no FPL submit.

When that looks right:

```bash
# in .env
SUGGEST_TRANSFERS=false
APPLY_TRANSFERS=true
docker compose run --rm pipeline
```

Learn then apply in the same container:

```bash
RUN_LEARNING=1
PREMIERLEARNING_ARGS=--horizon-reps 5
APPLY_TRANSFERS=true
```

Credentials can still fall back to `.vscode/fantasy_login.json` if you run **outside** Docker with no env vars. Docker/AWS must use `FPL_EMAIL` / `FPL_PASSWORD`.

## 3. Phase 0 — first AWS run (manual)

Install [Terraform](https://developer.hashicorp.com/terraform/install) ≥ 1.3.

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# optional: billing_email = "you@example.com"
terraform init
terraform apply
```

Keep `enable_schedule = false`, `enable_split = false`, `apply_transfers = false` for the first apply. That creates ECR, an ECS cluster, a Fargate task definition, a CloudWatch log group, and an empty FPL secret.

Push the image (from the repo root; Docker must be running):

```bash
bash infra/scripts/push-image.sh
```

Put FPL credentials in Secrets Manager (values are not printed):

```bash
export FPL_EMAIL='you@example.com'
export FPL_PASSWORD='...'
# optional: FPL_REFRESH_TOKEN, FPL_ID
bash infra/scripts/put-fpl-secret.sh
```

Run one task. First run is **suggest only** (`apply_transfers = false` in tfvars):

```bash
bash infra/scripts/run-task.sh pipeline
```

Watch logs:

```bash
aws logs tail /ecs/premierlearning --follow --region eu-west-2
```

When suggest looks right, set `apply_transfers = true` in `terraform.tfvars`, `terraform apply`, and run the task again.

Need a sqlite file in the image for apply-only: have `database/fantasy.db` on disk **before** `push-image.sh` so `COPY database` includes it. Or set `run_learning = true` so the task trains first (use `pipeline_cpu = "4096"` and `pipeline_memory = "16384"`).

Idle Fargate costs nothing. Delete later with `terraform destroy` if you want the account empty.

## 4. Daily 06:00 UK schedule

Same one task, now every morning. In `terraform.tfvars`:

```hcl
enable_schedule = true
run_learning    = true
apply_transfers = false
pipeline_cpu    = "4096"
pipeline_memory = "16384"
schedule_expression = "cron(0 6 * * ? *)"
schedule_timezone   = "Europe/London"
notify_email        = "haljsmith97+fpl@gmail.com"
ses_from_email      = "haljsmith97@gmail.com"
```

```bash
terraform apply
```

The task **suggests** transfers and emails chip + transfer strategy. On FPL **transfer deadline day** (the UK calendar day of the next deadline, while that deadline is still ahead) it **applies** those transfers instead. `APPLY_ON_DEADLINE_DAY` is set on the task; leave `apply_transfers = false` so ordinary mornings stay suggest-only.

Emails go to the plus-alias so you can add a Gmail filter: **To** `haljsmith97+fpl@gmail.com` → **Never send it to Spam**. Click the SES verify link sent to that alias once.

The combined task still runs **learn then squadtools** in one container (sqlite on the task disk). After GW1 you can set `apply_within_hours` (e.g. `"36"`) so a stray apply cannot submit far from the deadline; leave it empty if you want 6am deadline-day applies when the deadline is later that afternoon.

## 5. Phase 2 — split learn and apply

```hcl
enable_split    = true
enable_schedule = true
apply_transfers = true
```

```bash
terraform apply
```

This adds:

- RDS Postgres `db.t4g.micro` (reachable from the ECS security group)
- SQS queue `premierlearning-learning-done`
- Fargate task **learn** (`RUN_LEARNING=1`, persist, publish SQS)
- Fargate task **apply** (reads Postgres, logs into FPL)
- Lambda that `RunTask`s apply when the queue message arrives

The app reads `DATABASE_URL` (Postgres) or `DATABASE_PATH` / `DATABASE_URL=sqlite:///...` (local).

RDS is in the default VPC and publicly addressable **only** from the ECS security group, so we can skip a NAT gateway. Turn `enable_split` off and destroy if you do not want the ~$12–15/month micro database.

## Useful commands

| What | Command |
| --- | --- |
| Who am I? | `aws sts get-caller-identity` |
| Push image | `bash infra/scripts/push-image.sh` |
| Update FPL secret | `bash infra/scripts/put-fpl-secret.sh` |
| Manual pipeline | `bash infra/scripts/run-task.sh pipeline` |
| Manual learn (phase 2) | `bash infra/scripts/run-task.sh learn` |
| Manual apply (phase 2) | `bash infra/scripts/run-task.sh apply` |
| Logs | `aws logs tail /ecs/premierlearning --follow --region eu-west-2` |
| Tear down | `cd infra/terraform && terraform destroy` |

## Cost notes

- Phase 0 with no running task: ECR storage + Secrets Manager (~$0.40/secret/month) + log storage.
- One apply task (1 vCPU, 15 minutes): cents.
- A full learning task (4 vCPU / 16 GB, ~1 hour): a few dollars.
- Phase 2 RDS `db.t4g.micro`: the main always-on cost. Destroy it when you are not using it.
- Do not create an EKS cluster for this project until the pipeline has applied transfers successfully.
