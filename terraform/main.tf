#module "s3" {
 # source         = "./modules/s3"
  #s3_bucket_name = var.s3_bucket_name
#}

#module "sagemaker" {
 # source             = "./modules/sagemaker"
  #sagemaker_instance_name = var.sagemaker_instance_name
  #sagemaker_instance_type = var.sagemaker_instance_type
#}

#module "sns" {
 # source = "./modules/sns"
  #sns_topic_name = var.sns_topic_name
#}

#module "cloudwatch" {
 # source = "./modules/cloudwatch"
  #cloudwatch_log_group_name = var.cloudwatch_log_group_name
#}

#module "kinesis" {
 # source = "./modules/kinesis"
  #kinesis_stream_name      = var.kinesis_stream_name
  #kinesis_shard_count      = var.kinesis_shard_count
  #kinesis_retention_period = var.kinesis_retention_period
#}

#module "eventbridge" {
 # source = "./modules/eventbridge"
  #event_rule_name        = var.event_rule_name
  #event_rule_description = var.event_rule_description
#}


#module "cloudtrail" {
 # source       = "./modules/cloudtrail"
  #cloudtrail_name = var.cloudtrail_name
#}

# MODULES ORCHESTRATOR

module "network" {
    source               = "./modules/network"
    vpc_cidr             = "${var.vpc_cidr}"
    vpc_az1              = "${var.vpc_az1}"
    vpc_az2              = "${var.vpc_az2}"
    vpc_sn_pub_az1_cidr  = "${var.vpc_sn_pub_az1_cidr}"
    vpc_sn_pub_az2_cidr  = "${var.vpc_sn_pub_az2_cidr}"
    vpc_sn_priv_az1_cidr = "${var.vpc_sn_priv_az1_cidr}"
    vpc_sn_priv_az2_cidr = "${var.vpc_sn_priv_az2_cidr}"
}

module "compute" {
    source                   = "./modules/compute"
    ec2_lt_name              = "${var.ec2_lt_name}"
    ec2_lt_ami               = "${var.ec2_lt_ami}"
    ec2_lt_instance_type     = "${var.ec2_lt_instance_type}"
    ec2_lt_ssh_key_name      = "${var.ec2_lt_ssh_key_name}"
    ec2_lb_name              = "${var.ec2_lb_name}"
    ec2_lb_tg_name           = "${var.ec2_lb_tg_name}"
    ec2_asg_name             = "${var.ec2_asg_name}"
    ec2_asg_desired_capacity = "${var.ec2_asg_desired_capacity}"
    ec2_asg_min_size         = "${var.ec2_asg_min_size}"
    ec2_asg_max_size         = "${var.ec2_asg_max_size}"
    vpc_cidr                 = "${var.vpc_cidr}"
    vpc_id                   = "${module.network.vpc_id}"
    vpc_sn_pub_az1_id        = "${module.network.vpc_sn_pub_az1_id}"
    vpc_sn_pub_az2_id        = "${module.network.vpc_sn_pub_az2_id}"
    vpc_sg_pub_id            = "${module.network.vpc_sg_pub_id}"
}



