resource "aws_cloudwatch_event_rule" "weekly" {
  count               = var.enable_schedule ? 1 : 0
  name                = "${var.name_prefix}-weekly"
  description         = "Start the learn-then-apply pipeline"
  schedule_expression = var.schedule_expression
}

resource "aws_cloudwatch_event_target" "weekly" {
  count     = var.enable_schedule ? 1 : 0
  rule      = aws_cloudwatch_event_rule.weekly[0].name
  target_id = "pipeline"
  arn       = aws_ecs_cluster.this.arn
  role_arn  = aws_iam_role.events[0].arn

  ecs_target {
    task_count          = 1
    task_definition_arn = var.enable_split ? aws_ecs_task_definition.learn[0].arn : aws_ecs_task_definition.pipeline.arn
    launch_type         = "FARGATE"
    platform_version    = "LATEST"
    network_configuration {
      subnets          = aws_subnet.private[*].id
      security_groups  = [aws_security_group.ecs.id]
      assign_public_ip = false
    }
  }
}
