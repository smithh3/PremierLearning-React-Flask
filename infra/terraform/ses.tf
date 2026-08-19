resource "aws_ses_email_identity" "notify" {
  count = var.notify_email == "" ? 0 : 1
  email = var.notify_email
}
