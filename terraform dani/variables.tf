# NETWORK VARS CUSTOM VALUES

variable "vpc_cidr" {
    type    = string
    default = "10.0.0.0/16"
}

variable "vpc_az1" {
    type    = string
    default = "us-east-1a"
}

variable "vpc_az2" {
    type    = string
    default = "us-east-1c"
}

variable "vpc_sn_pub_az1_cidr" {
    type    = string
    default = "10.0.1.0/24"
}

variable "vpc_sn_pub_az2_cidr" {
    type    = string
    default = "10.0.2.0/24"
}

variable "vpc_sn_priv_az1_cidr" {
    type    = string
    default = "10.0.3.0/24"
}

variable "vpc_sn_priv_az2_cidr" {
    type    = string
    default = "10.0.4.0/24"
}

# COMPUTE VARS CUSTOM VALUES

variable "ec2_ch_name" {
    type    = string
    default = "EC2 Challenge"
}

variable "ec2_ami_id" {
    type    = string
    default = "ami-0123456789abcdef0"  
}

variable "ec2_instance_type" {
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

variable "ec2_desired_capacity" {
    type    = number
    default = 4
}

variable "ec2_min_size" {
    type    = number
    default = 2
}

variable "ec2_max_size" {
    type    = number
    default = 8
}
