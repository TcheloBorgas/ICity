resource "aws_security_group" "load_balancer_sg" {
  name_prefix = "load-balancer-sg-"
}

resource "aws_security_group_rule" "lb_http_ingress" {
  type        = "ingress"
  from_port   = 80
  to_port     = 80
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]
  security_group_id = aws_security_group.load_balancer_sg.id
}

resource "aws_security_group_rule" "lb_https_ingress" {
  type        = "ingress"
  from_port   = 443
  to_port     = 443
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]
  security_group_id = aws_security_group.load_balancer_sg.id
}

resource "aws_lb" "load_balancer" {
  name               = "my-load-balancer"
  internal           = false
  load_balancer_type = "application"
  security_groups   = [aws_security_group.load_balancer_sg.id]
  enable_deletion_protection = false
}

resource "aws_lb_target_group" "target_group" {
  name     = "my-target-group"
  port     = 80
  protocol = "HTTP"
  vpc_id   = var.vpc_id
}

resource "aws_lb_listener" "listener" {
  load_balancer_arn = aws_lb.load_balancer.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type = "fixed-response"

    fixed_response {
      content_type = "text/plain"
      message_body = "Hello, world!"
      status_code  = "200"
    }
  }
}

resource "aws_autoscaling_group" "autoscaling_group" {
  name                 = "my-autoscaling-group"
  desired_capacity    = var.ec2_desired_capacity
  max_size            = var.ec2_max_size
  min_size            = var.ec2_min_size
  health_check_grace_period = 300
  health_check_type   = "ELB"

  launch_configuration = aws_launch_configuration.launch_config.name

  tag {
    key                 = "Name"
    value               = "EC2 Challenge"
    propagate_at_launch = true
  }

  vpc_zone_identifier = var.subnet_ids
}

resource "aws_launch_configuration" "launch_config" {
  name_prefix = "my-launch-config-"
  image_id      = var.ec2_ami_id
  instance_type = var.ec2_instance_type

  security_groups = [aws_security_group.load_balancer_sg.id]
}
