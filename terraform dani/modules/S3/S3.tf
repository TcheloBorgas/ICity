resource "aws_s3_bucket" "image_bucket" {
  bucket = var.s3_bucket_name
  acl    = "private"

  lifecycle_rule {
    id      = "image_retention"
    status  = "Enabled"

  }
}
