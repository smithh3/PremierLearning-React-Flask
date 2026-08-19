output "region" {
  value = var.aws_region
}

output "ecr_repository_url" {
  value = aws_ecr_repository.pipeline.repository_url
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "pipeline_task_definition" {
  value = aws_ecs_task_definition.pipeline.arn
}

output "fpl_secret_arn" {
  value = aws_secretsmanager_secret.fpl.arn
}

output "log_group" {
  value = aws_cloudwatch_log_group.pipeline.name
}

output "ecs_security_group_id" {
  value = aws_security_group.ecs.id
}

output "subnet_ids" {
  value       = aws_subnet.private[*].id
  description = "Private subnets. Fargate has no public IP; traffic exits via the NAT instance."
}

output "assign_public_ip" {
  value = "DISABLED"
}

output "nat_public_ip" {
  value       = aws_instance.nat.public_ip
  description = "Observational only. Fargate does not use this value; it discovers egress IP at runtime. Changes if the NAT instance is stopped and started."
}

output "learn_task_definition" {
  value = var.enable_split ? aws_ecs_task_definition.learn[0].arn : null
}

output "apply_task_definition" {
  value = var.enable_split ? aws_ecs_task_definition.apply[0].arn : null
}

output "learning_done_queue_url" {
  value = var.enable_split ? aws_sqs_queue.learning_done[0].url : null
}
