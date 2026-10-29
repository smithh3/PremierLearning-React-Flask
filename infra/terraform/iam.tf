data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${var.name_prefix}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "execution_secrets" {
  statement {
    actions = [
      "secretsmanager:GetSecretValue",
    ]
    resources = concat(
      [aws_secretsmanager_secret.fpl.arn],
      aws_secretsmanager_secret.db[*].arn,
    )
  }
}

resource "aws_iam_role_policy" "execution_secrets" {
  name   = "secrets"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution_secrets.json
}

resource "aws_iam_role" "task" {
  name               = "${var.name_prefix}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

data "aws_iam_policy_document" "task" {
  statement {
    actions = [
      "secretsmanager:GetSecretValue",
      "secretsmanager:PutSecretValue",
    ]
    resources = [aws_secretsmanager_secret.fpl.arn]
  }

  dynamic "statement" {
    for_each = var.notify_email != "" || var.ses_from_email != "" ? [1] : []
    content {
      actions   = ["ses:SendEmail", "ses:SendRawEmail"]
      resources = ["*"]
    }
  }

  dynamic "statement" {
    for_each = var.enable_split ? [1] : []
    content {
      actions   = ["sqs:SendMessage"]
      resources = aws_sqs_queue.learning_done[*].arn
    }
  }

  dynamic "statement" {
    for_each = var.enable_schedule ? [1] : []
    content {
      actions = [
        "scheduler:GetSchedule",
        "scheduler:UpdateSchedule",
      ]
      resources = [
        "arn:aws:scheduler:${var.aws_region}:${data.aws_caller_identity.current.account_id}:schedule/default/${var.name_prefix}-daily",
      ]
    }
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "runtime"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}

data "aws_iam_policy_document" "events_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com", "scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "events" {
  count              = var.enable_schedule ? 1 : 0
  name               = "${var.name_prefix}-events"
  assume_role_policy = data.aws_iam_policy_document.events_assume.json
}

data "aws_iam_policy_document" "events" {
  count = var.enable_schedule ? 1 : 0
  statement {
    actions = ["ecs:RunTask"]
    resources = concat(
      ["${aws_ecs_task_definition.pipeline.arn_without_revision}:*"],
      [for arn in aws_ecs_task_definition.learn[*].arn_without_revision : "${arn}:*"],
    )
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
    actions   = ["ecs:TagResource"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "events" {
  count  = var.enable_schedule ? 1 : 0
  name   = "run-task"
  role   = aws_iam_role.events[0].id
  policy = data.aws_iam_policy_document.events[0].json
}
