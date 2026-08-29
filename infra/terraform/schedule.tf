resource "aws_scheduler_schedule" "daily" {
  count = var.enable_schedule ? 1 : 0
  name  = "${var.name_prefix}-daily"

  description                = "06:00 UK: suggest transfers, or apply on FPL deadline day"
  schedule_expression          = var.schedule_expression
  schedule_expression_timezone = var.schedule_timezone
  state                        = "ENABLED"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_ecs_cluster.this.arn
    role_arn = aws_iam_role.events[0].arn

    ecs_parameters {
      task_definition_arn = var.enable_split ? aws_ecs_task_definition.learn[0].arn : aws_ecs_task_definition.pipeline.arn
      launch_type         = "FARGATE"
      task_count          = 1
      platform_version    = "LATEST"

      network_configuration {
        subnets          = aws_subnet.private[*].id
        security_groups  = [aws_security_group.ecs.id]
        assign_public_ip = false
      }
    }
  }
}
