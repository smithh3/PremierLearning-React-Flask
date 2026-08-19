resource "aws_ecr_repository" "pipeline" {
  name                 = var.name_prefix
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_cloudwatch_log_group" "pipeline" {
  name              = "/ecs/${var.name_prefix}"
  retention_in_days = 30
}

resource "aws_secretsmanager_secret" "fpl" {
  name                    = "${var.name_prefix}/fpl"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "fpl" {
  secret_id = aws_secretsmanager_secret.fpl.id
  secret_string = jsonencode({
    EMAIL         = ""
    PASSWORD      = ""
    REFRESH_TOKEN = ""
    ID            = ""
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "aws_security_group" "ecs" {
  name        = "${var.name_prefix}-ecs"
  description = "Fargate tasks: outbound HTTPS to FPL, optional Postgres to RDS"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "All outbound; internet is via the NAT instance, not a Fargate public IP"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.name_prefix}-ecs"
  }
}

resource "aws_ecs_cluster" "this" {
  name = var.name_prefix

  setting {
    name  = "containerInsights"
    value = "disabled"
  }
}

resource "aws_cloudwatch_log_metric_filter" "failures" {
  name           = "${var.name_prefix}-failures"
  log_group_name = aws_cloudwatch_log_group.pipeline.name
  pattern        = "?ERROR ?Traceback ?\"FPL login failed\" ?\"SystemExit\""

  metric_transformation {
    name      = "PremierLearningFailures"
    namespace = "PremierLearning"
    value     = "1"
  }
}

resource "aws_cloudwatch_metric_alarm" "failures" {
  alarm_name          = "${var.name_prefix}-task-errors"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "PremierLearningFailures"
  namespace           = "PremierLearning"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "A premierlearning/squadtools task logged an error"
  treat_missing_data  = "notBreaching"
}
