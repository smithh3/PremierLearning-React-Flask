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
  description = "EventBridge Scheduler that starts the pipeline two hours before the next FPL deadline. Each run books the one after."
}

variable "schedule_expression" {
  type        = string
  default     = "at(2026-10-31T09:00:00)"
  description = "EventBridge Scheduler expression for the next run. The task replaces this with the following deadline minus two hours."
}

variable "schedule_hours_before_deadline" {
  type        = number
  default     = 5
  description = "Hours before each FPL deadline the scheduled run starts. The learning_v2 blend trains for about 2.5 hours on 4 vCPU, so 2 was too late to apply before the deadline."
}

variable "schedule_timezone" {
  type        = string
  default     = "Europe/London"
  description = "IANA timezone for schedule_expression."
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
  description = "Extra argv for premierlearning. The model is the learning_v2 blend the fantasy manager evaluation scored best (75% v2_log with a minutes-bands head, 25% v2_log with a 5-GW net) unless these args choose another: --v1 for the old model (e.g. \"--v1 --horizon-reps 5\"), or a v1 recipe such as --h1-poisson. v1-only flags like --horizon-reps are ignored by the blend. The pipeline always prepends --update so Fargate pulls live FPL injury/chance news; pass --no-update to skip."
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
  default     = "haljsmith97+fpl@gmail.com"
  description = "SES destination for the Fargate predictions email (HTML table in the body). Leave empty to disable."
}

variable "ses_from_email" {
  type        = string
  default     = "haljsmith97@gmail.com"
  description = "Verified SES From address. Keep this as the base Gmail identity; plus-aliases go in notify_email."
}
