locals {
  ses_from = var.ses_from_email != "" ? var.ses_from_email : var.notify_email
  ses_identities = distinct([
    for address in [var.notify_email, local.ses_from] : address if address != ""
  ])
}

resource "aws_ses_email_identity" "notify" {
  for_each = toset(local.ses_identities)
  email    = each.value
}

moved {
  from = aws_ses_email_identity.notify[0]
  to   = aws_ses_email_identity.notify["haljsmith97@gmail.com"]
}
