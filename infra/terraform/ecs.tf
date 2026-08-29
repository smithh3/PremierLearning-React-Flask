locals {
  image = "${aws_ecr_repository.pipeline.repository_url}:${var.image_tag}"

  fpl_secrets = [
    {
      name      = "FPL_EMAIL"
      valueFrom = "${aws_secretsmanager_secret.fpl.arn}:EMAIL::"
    },
    {
      name      = "FPL_PASSWORD"
      valueFrom = "${aws_secretsmanager_secret.fpl.arn}:PASSWORD::"
    },
    {
      name      = "FPL_REFRESH_TOKEN"
      valueFrom = "${aws_secretsmanager_secret.fpl.arn}:REFRESH_TOKEN::"
    },
    {
      name      = "FPL_ID"
      valueFrom = "${aws_secretsmanager_secret.fpl.arn}:ID::"
    },
  ]

  pipeline_env = [
    { name = "FPL_LOGIN_HEADLESS", value = "1" },
    { name = "FPL_REFRESH_SECRET_ARN", value = aws_secretsmanager_secret.fpl.arn },
    { name = "DATABASE_PATH", value = "/app/database/fantasy.db" },
    { name = "RUN_LEARNING", value = var.run_learning ? "1" : "0" },
    { name = "APPLY_TRANSFERS", value = var.apply_transfers ? "true" : "false" },
    { name = "SUGGEST_TRANSFERS", value = var.apply_transfers ? "false" : "true" },
    { name = "APPLY_ON_DEADLINE_DAY", value = "true" },
    { name = "PREMIERLEARNING_ARGS", value = var.premierlearning_args },
    { name = "APPLY_WITHIN_HOURS", value = var.apply_within_hours },
    { name = "AWS_DEFAULT_REGION", value = var.aws_region },
    { name = "NOTIFY_EMAIL", value = var.notify_email },
    { name = "SES_FROM_EMAIL", value = var.ses_from_email != "" ? var.ses_from_email : var.notify_email },
  ]
}

resource "aws_ecs_task_definition" "pipeline" {
  family                   = "${var.name_prefix}-pipeline"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.pipeline_cpu
  memory                   = var.pipeline_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "pipeline"
      image     = local.image
      essential = true
      environment = local.pipeline_env
      secrets     = local.fpl_secrets
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.pipeline.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "pipeline"
        }
      }
    }
  ])
}
