variable "aws_region" {
  type        = string
  default     = "eu-west-2"
  description = "Use London and keep every resource in this region."
}

variable "name_prefix" {
  type    = string
  default = "premierlearning"
}

variable "image_tag" {
  type        = string
  default     = "latest"
  description = "ECR image tag to run. Push the image before the first ECS run."
}

variable "pipeline_cpu" {
  type        = string
  default     = "1024"
  description = "Fargate CPU units. Use 4096 for a full learning run."
}

variable "pipeline_memory" {
  type        = string
  default     = "2048"
  description = "Fargate memory (MB). Use 16384 for a full learning run."
}

variable "enable_schedule" {
  type        = bool
  default     = false
  description = "Phase 1: weekly EventBridge rule that starts the pipeline task."
}

variable "schedule_expression" {
  type        = string
  default     = "cron(0 8 ? * FRI *)"
  description = "EventBridge cron in UTC. Friday 08:00 UTC is 09:00 UK in BST."
}

variable "enable_split" {
  type        = bool
  default     = false
  description = "Phase 2: RDS Postgres, SQS, separate learn + apply tasks."
}

variable "apply_transfers" {
  type        = bool
  default     = false
  description = "When true, the pipeline/apply task submits transfers to FPL."
}

variable "run_learning" {
  type        = bool
  default     = false
  description = "When true, the combined pipeline task trains before squadtools."
}

variable "premierlearning_args" {
  type        = string
  default     = ""
  description = "Extra argv for premierlearning. Empty is a full train. Use --horizon-only --horizon-reps 5 for a shorter run."
}

variable "apply_within_hours" {
  type        = string
  default     = ""
  description = "Optional APPLY_WITHIN_HOURS guard for live submits."
}

variable "nat_instance_type" {
  type        = string
  default     = "t3.micro"
  description = "Always-on NAT EC2 that Fargate uses for internet (FPL sees this address, not Fargate). t3.micro is Free Tier eligible."
}

variable "billing_email" {
  type        = string
  default     = ""
  description = "If set, create a $10 estimated-charges alarm emailed here. Enable Receive Billing Alerts in the Billing console first."
}

variable "notify_email" {
  type        = string
  default     = "haljsmith97@gmail.com"
  description = "SES destination for the Fargate predictions email (HTML table in the body). Leave empty to disable."
}
