variable "sagemaker_instance_name" {
  description = "Name of the SageMaker Notebook instance"
}

variable "sagemaker_instance_type" {
  description = "Instance type for the SageMaker Notebook instance"
  default     = "ml.t2.medium"
}
