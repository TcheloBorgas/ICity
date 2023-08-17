provider "aws" {
  region                   = "us-east-1"
  shared_config_files      = ["./.aws/config"]
  shared_credentials_files = ["./.aws/credentials"]
  profile                  = "ICity"
}

terraform {
  required_version = "~> 1.0.5"
}

required_providers {
  aws = {
    source  = "hashicorp/aws"
    version = "~> 3.63.0"
  }
}

# Bucket S3 para armazenar imagens
resource "aws_s3_bucket" "image_bucket" {
  bucket = "my-image-bucket"
  acl    = "private"

  lifecycle_rule {
    id      = "image_retention"
    status  = "Enabled"

    expiration {
      days = 20
    }
  }
}

# Instância EC2 para hospedar a API
resource "aws_instance" "api_instance" {
  ami           = "ami-0c55b159cbfafe1f0" # Altere para a AMI desejada
  instance_type = "t2.micro"

  tags = {
    Name = "API_Instance"
  }
}

# Configuração SageMaker Notebook
resource "aws_sagemaker_notebook_instance" "sagemaker_instance" {
  name          = "my-sagemaker-instance"
  role_arn      = aws_iam_role.sagemaker_role.arn
  instance_type = "ml.t2.medium"
}

resource "aws_iam_role" "sagemaker_role" {
  name = "SageMakerExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Principal = {
          Service = "sagemaker.amazonaws.com"
        },
        Effect = "Allow",
        Sid    = ""
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3_access" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
  role       = aws_iam_role.sagemaker_role.name
}

# Configuração do Amazon SNS
resource "aws_sns_topic" "sns_topic" {
  name = "my-sns-topic"
}

# Configuração do AWS CloudWatch
resource "aws_cloudwatch_log_group" "cloudwatch_log_group" {
  name = "my-cloudwatch-log-group"
}

# Configuração do AWS Lambda
resource "aws_lambda_function" "lambda_function" {
  function_name = "my-lambda-function"
  handler       = "index.handler"
  runtime       = "nodejs12.x"
  role          = aws_iam_role.lambda_role.arn

  filename = "lambda_function_payload.zip"
}

resource "aws_iam_role" "lambda_role" {
  name = "LambdaExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Principal = {
          Service = "lambda.amazonaws.com"
        },
        Effect = "Allow",
        Sid    = ""
      }
    ]
  })
}

# Configuração do Amazon Kinesis
resource "aws_kinesis_stream" "kinesis_stream" {
  name             = "my-kinesis-stream"
  shard_count      = 1
  retention_period = 24

  shard_level_metrics = [
    "IncomingBytes",
    "OutgoingBytes",
  ]
}

# Configuração do Amazon EventBridge
resource "aws_eventbridge_rule" "event_rule" {
  name        = "my-event-rule"
  description = "Rule for triggering events"
  event_pattern = jsonencode({
    source = ["aws.sagemaker"]
  })

  target {
    arn = aws_lambda_function.lambda_function.arn
    id  = "lambda_target"
  }
}

# Configuração do Amazon API Gateway
resource "aws_api_gateway_rest_api" "api_gateway" {
  name        = "my-api-gateway"
  description = "API Gateway for image processing"
}

resource "aws_api_gateway_resource" "api_resource" {
  rest_api_id = aws_api_gateway_rest_api.api_gateway.id
  parent_id   = aws_api_gateway_rest_api.api_gateway.root_resource_id
  path_part   = "images"
}

resource "aws_api_gateway_method" "api_method" {
  rest_api_id   = aws_api_gateway_rest_api.api_gateway.id
  resource_id   = aws_api_gateway_resource.api_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "api_integration" {
  rest_api_id = aws_api_gateway_rest_api.api_gateway.id
  resource_id = aws_api_gateway_resource.api_resource.id
  http_method = aws_api_gateway_method.api_method.http_method
  type        = "AWS_PROXY"
  uri         = aws_lambda_function.lambda_function.invoke_arn
}

resource "aws_api_gateway_method_response" "api_response" {
  rest_api_id = aws_api_gateway_rest_api.api_gateway.id
  resource_id = aws_api_gateway_resource.api_resource.id
  http_method = aws_api_gateway_method.api_method.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = "'*'"
  }
}

resource "aws_api_gateway_integration_response" "api_integration_response" {
  rest_api_id = aws_api_gateway_rest_api.api_gateway.id
  resource_id = aws_api_gateway_resource.api_resource.id
  http_method = aws_api_gateway_method.api_method.http_method
  status_code = aws_api_gateway_method_response.api_response.status_code

  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = "'*'"
  }

  response_templates = {
    "application/json" = ""
  }
}

# Configuração do AWS CloudTrail
resource "aws_cloudtrail" "cloudtrail" {
  name                          = "my-cloudtrail"
  s3_bucket_name                = aws_s3_bucket.image_bucket.id
  include_global_service_events = true
}
