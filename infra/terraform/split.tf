resource "random_password" "db" {
  count   = var.enable_split ? 1 : 0
  length  = 20
  special = false
}

resource "aws_secretsmanager_secret" "db" {
  count                   = var.enable_split ? 1 : 0
  name                    = "${var.name_prefix}/db"
  recovery_window_in_days = 0
}

resource "aws_db_subnet_group" "this" {
  count      = var.enable_split ? 1 : 0
  name       = var.name_prefix
  subnet_ids = data.aws_subnets.default.ids
}

resource "aws_security_group" "rds" {
  count       = var.enable_split ? 1 : 0
  name        = "${var.name_prefix}-rds"
  description = "Postgres from ECS tasks only"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "this" {
  count                      = var.enable_split ? 1 : 0
  identifier                 = var.name_prefix
  engine                     = "postgres"
  engine_version             = "16"
  instance_class             = "db.t4g.micro"
  allocated_storage          = 20
  db_name                    = "fantasy"
  username                   = "premierlearning"
  password                   = random_password.db[0].result
  db_subnet_group_name       = aws_db_subnet_group.this[0].name
  vpc_security_group_ids     = [aws_security_group.rds[0].id]
  publicly_accessible        = true
  skip_final_snapshot        = true
  deletion_protection        = false
  backup_retention_period    = 1
  auto_minor_version_upgrade = true
}

resource "aws_secretsmanager_secret_version" "db" {
  count     = var.enable_split ? 1 : 0
  secret_id = aws_secretsmanager_secret.db[0].id
  secret_string = jsonencode({
    DATABASE_URL = "postgresql://premierlearning:${random_password.db[0].result}@${aws_db_instance.this[0].address}:5432/fantasy"
  })
}

resource "aws_sqs_queue" "learning_done" {
  count                      = var.enable_split ? 1 : 0
  name                       = "${var.name_prefix}-learning-done"
  visibility_timeout_seconds = 120
  message_retention_seconds  = 86400
}

resource "aws_ecs_task_definition" "learn" {
  count                    = var.enable_split ? 1 : 0
  family                   = "${var.name_prefix}-learn"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "4096"
  memory                   = "16384"
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "learn"
      image     = local.image
      essential = true
      environment = [
        { name = "RUN_LEARNING", value = "1" },
        { name = "SKIP_SQUADTOOLS", value = "1" },
        { name = "PREMIERLEARNING_ARGS", value = var.premierlearning_args },
        { name = "LEARNING_DONE_QUEUE_URL", value = aws_sqs_queue.learning_done[0].url },
        { name = "FPL_LOGIN_HEADLESS", value = "1" },
      ]
      secrets = [
        {
          name      = "DATABASE_URL"
          valueFrom = "${aws_secretsmanager_secret.db[0].arn}:DATABASE_URL::"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.pipeline.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "learn"
        }
      }
    }
  ])
}

resource "aws_ecs_task_definition" "apply" {
  count                    = var.enable_split ? 1 : 0
  family                   = "${var.name_prefix}-apply"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "apply"
      image     = local.image
      essential = true
      environment = concat(
        [
          { name = "RUN_LEARNING", value = "0" },
          { name = "APPLY_TRANSFERS", value = var.apply_transfers ? "true" : "false" },
          { name = "SUGGEST_TRANSFERS", value = var.apply_transfers ? "false" : "true" },
          { name = "APPLY_ON_DEADLINE_DAY", value = "true" },
          { name = "APPLY_WITHIN_HOURS", value = var.apply_within_hours },
          { name = "FPL_LOGIN_HEADLESS", value = "1" },
          { name = "FPL_REFRESH_SECRET_ARN", value = aws_secretsmanager_secret.fpl.arn },
          { name = "AWS_DEFAULT_REGION", value = var.aws_region },
          { name = "NOTIFY_EMAIL", value = var.notify_email },
          { name = "SES_FROM_EMAIL", value = var.ses_from_email != "" ? var.ses_from_email : var.notify_email },
        ]
      )
      secrets = concat(
        local.fpl_secrets,
        [
          {
            name      = "DATABASE_URL"
            valueFrom = "${aws_secretsmanager_secret.db[0].arn}:DATABASE_URL::"
          }
        ]
      )
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.pipeline.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "apply"
        }
      }
    }
  ])
}

data "archive_file" "run_apply" {
  count       = var.enable_split ? 1 : 0
  type        = "zip"
  source_file = "${path.module}/lambda/run_apply.py"
  output_path = "${path.module}/lambda/run_apply.zip"
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "run_apply" {
  count              = var.enable_split ? 1 : 0
  name               = "${var.name_prefix}-run-apply"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "run_apply_logs" {
  count      = var.enable_split ? 1 : 0
  role       = aws_iam_role.run_apply[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "run_apply" {
  count = var.enable_split ? 1 : 0
  statement {
    actions   = ["ecs:RunTask"]
    resources = ["${aws_ecs_task_definition.apply[0].arn_without_revision}:*"]
    condition {
      test     = "ArnEquals"
      variable = "ecs:cluster"
      values   = [aws_ecs_cluster.this.arn]
    }
  }
  statement {
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.execution.arn, aws_iam_role.task.arn]
  }
  statement {
    actions = [
      "sqs:ReceiveMessage",
      "sqs:DeleteMessage",
      "sqs:GetQueueAttributes",
      "sqs:ChangeMessageVisibility",
    ]
    resources = [aws_sqs_queue.learning_done[0].arn]
  }
}

resource "aws_iam_role_policy" "run_apply" {
  count  = var.enable_split ? 1 : 0
  name   = "run-apply-task"
  role   = aws_iam_role.run_apply[0].id
  policy = data.aws_iam_policy_document.run_apply[0].json
}

resource "aws_lambda_function" "run_apply" {
  count            = var.enable_split ? 1 : 0
  function_name    = "${var.name_prefix}-run-apply"
  filename         = data.archive_file.run_apply[0].output_path
  source_code_hash = data.archive_file.run_apply[0].output_base64sha256
  handler          = "run_apply.handler"
  runtime          = "python3.12"
  timeout          = 30
  role             = aws_iam_role.run_apply[0].arn

  environment {
    variables = {
      CLUSTER          = aws_ecs_cluster.this.name
      TASK_DEFINITION  = aws_ecs_task_definition.apply[0].arn
      SUBNETS          = join(",", aws_subnet.private[*].id)
      SECURITY_GROUPS  = aws_security_group.ecs.id
    }
  }
}

resource "aws_lambda_event_source_mapping" "learning_done" {
  count            = var.enable_split ? 1 : 0
  event_source_arn = aws_sqs_queue.learning_done[0].arn
  function_name    = aws_lambda_function.run_apply[0].arn
  batch_size       = 1
}
