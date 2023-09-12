#resource "aws_eventbridge_rule" "event_rule" {
 # name        = var.event_rule_name
  #description = var.event_rule_description

  #event_pattern = jsonencode({
   # source = ["aws.sagemaker"]
  #}

  #target {
   # arn = aws_lambda_function.lambda_function.arn
    #id  = "lambda_target"
  #}
#}
