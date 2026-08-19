resource "aws_sns_topic" "billing" {
  count    = var.billing_email == "" ? 0 : 1
  provider = aws.us_east_1
  name     = "${var.name_prefix}-billing"
}

resource "aws_sns_topic_subscription" "billing" {
  count     = var.billing_email == "" ? 0 : 1
  provider  = aws.us_east_1
  topic_arn = aws_sns_topic.billing[0].arn
  protocol  = "email"
  endpoint  = var.billing_email
}

resource "aws_cloudwatch_metric_alarm" "billing" {
  count               = var.billing_email == "" ? 0 : 1
  provider            = aws.us_east_1
  alarm_name          = "${var.name_prefix}-estimated-charges-10"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = 21600
  statistic           = "Maximum"
  threshold           = 10
  alarm_description   = "AWS estimated charges reached $10"
  alarm_actions       = [aws_sns_topic.billing[0].arn]
  treat_missing_data  = "notBreaching"

  dimensions = {
    Currency = "USD"
  }
}
