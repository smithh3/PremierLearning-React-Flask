#!/usr/bin/env bash
# Reads FPL_EMAIL / FPL_PASSWORD / FPL_REFRESH_TOKEN / FPL_ID from the environment
# and writes them into the Secrets Manager secret created by Terraform.
# Does not print the password.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${FPL_EMAIL:?Set FPL_EMAIL}"
: "${FPL_PASSWORD:?Set FPL_PASSWORD}"

cd "$ROOT/infra/terraform"
SECRET_ARN="$(terraform output -raw fpl_secret_arn)"

SECRET_JSON="$(python - <<'PY'
import json, os
print(json.dumps({
    "EMAIL": os.environ["FPL_EMAIL"],
    "PASSWORD": os.environ["FPL_PASSWORD"],
    "REFRESH_TOKEN": os.environ.get("FPL_REFRESH_TOKEN", ""),
    "ID": os.environ.get("FPL_ID", ""),
}))
PY
)"

aws secretsmanager put-secret-value \
  --secret-id "$SECRET_ARN" \
  --secret-string "$SECRET_JSON" \
  >/dev/null

echo "Updated FPL secret $SECRET_ARN (values not printed)"
