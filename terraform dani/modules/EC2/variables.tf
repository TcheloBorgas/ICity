
variable "ec2_desired_capacity" {
  description = "Desired capacity of the Auto Scaling Group"
  default     = 2
}

variable "ec2_min_size" {
  description = "Minimum size of the Auto Scaling Group"
  default     = 2
}

variable "ec2_max_size" {
  description = "Maximum size of the Auto Scaling Group"
  default     = 4
}

# Variables for EC2 instance
variable "ec2_ami_id" {
   type    = string
    default = "ami-0123456789abcdef0"
  
}

variable "instance_type" {
    type    = string
    default = "t2.micro"
}

variable "ec2_lb_name" {
    type    = string
    default = "ec2-lb-notifier"
}

variable "ec2_tg_name" {
    type    = string
    default = "ec2-lb-tg-notifier"
}

variable "ec2_asg_name" {
    type    = string
    default = "ec2-asg-notifier"
}

variable "ec2_ch_name" {
    type    = string
    default = "EC2 Challenge"
}
