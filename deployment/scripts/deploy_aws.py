import boto3
import paramiko
import time
import os
import logging
from botocore.exceptions import ClientError

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# AWS Configuration
AWS_REGION = "us-west-2"
KEY_PAIR_NAME = "aws-keypair"
INSTANCE_TYPE = "t2.medium"
AMI_ID = "ami-0abcdef1234567890"
S3_BUCKET_NAME = "sentiment-analysis-models"
MODEL_FILE_PATH = "models/sentiment_model.tar.gz"
DEPLOY_SCRIPT_PATH = "scripts/run_model_server.sh"
SECURITY_GROUP_NAME = "sentiment-analysis-sg"
SECURITY_GROUP_DESCRIPTION = "Security group for Sentiment Analysis Deployment"

# Initialize Boto3 clients
ec2 = boto3.client('ec2', region_name=AWS_REGION)
s3 = boto3.client('s3', region_name=AWS_REGION)

# Step 1: Upload model to S3
def upload_model_to_s3():
    try:
        logger.info(f"Uploading model to S3 bucket {S3_BUCKET_NAME}...")
        s3.upload_file(MODEL_FILE_PATH, S3_BUCKET_NAME, os.path.basename(MODEL_FILE_PATH))
        logger.info("Model uploaded successfully.")
    except ClientError as e:
        logger.error(f"Error uploading model to S3: {e}")
        raise

# Step 2: Create security group
def create_security_group():
    try:
        logger.info(f"Creating security group {SECURITY_GROUP_NAME}...")
        response = ec2.create_security_group(
            GroupName=SECURITY_GROUP_NAME,
            Description=SECURITY_GROUP_DESCRIPTION,
            VpcId=get_default_vpc_id()
        )
        security_group_id = response['GroupId']
        logger.info(f"Security group created with ID {security_group_id}")

        # Add rules to the security group
        ec2.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]  # Allow SSH from anywhere
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 80,
                    'ToPort': 80,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]  # Allow HTTP access
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 443,
                    'ToPort': 443,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]  # Allow HTTPS access
                }
            ]
        )
        logger.info(f"Ingress rules added to security group {security_group_id}")
        return security_group_id
    except ClientError as e:
        logger.error(f"Error creating security group: {e}")
        raise

# Step 3: Get the default VPC ID
def get_default_vpc_id():
    response = ec2.describe_vpcs()
    for vpc in response['Vpcs']:
        if vpc['IsDefault']:
            return vpc['VpcId']
    raise Exception("No default VPC found")

# Step 4: Launch EC2 instance
def launch_ec2_instance(security_group_id):
    try:
        logger.info("Launching EC2 instance...")
        instance = ec2.run_instances(
            ImageId=AMI_ID,
            InstanceType=INSTANCE_TYPE,
            KeyName=KEY_PAIR_NAME,
            MinCount=1,
            MaxCount=1,
            SecurityGroupIds=[security_group_id],
            TagSpecifications=[{
                'ResourceType': 'instance',
                'Tags': [{'Key': 'Name', 'Value': 'Sentiment-Analysis-Server'}]
            }]
        )
        instance_id = instance['Instances'][0]['InstanceId']
        logger.info(f"EC2 Instance {instance_id} launched.")

        # Wait for instance to be running
        waiter = ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])
        logger.info(f"Instance {instance_id} is now running.")
        
        # Retrieve public DNS
        instance_info = ec2.describe_instances(InstanceIds=[instance_id])
        public_dns = instance_info['Reservations'][0]['Instances'][0]['PublicDnsName']
        return instance_id, public_dns
    except ClientError as e:
        logger.error(f"Error launching EC2 instance: {e}")
        raise

# Step 5: Configure EC2 instance and deploy model
def configure_ec2_instance(public_dns):
    logger.info(f"Connecting to EC2 instance {public_dns}...")
    key = paramiko.RSAKey.from_private_key_file(f"./{KEY_PAIR_NAME}.pem")
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Wait for SSH to be available
    time.sleep(60)

    try:
        ssh_client.connect(hostname=public_dns, username="ec2-user", pkey=key)
        logger.info("Connected to EC2 instance via SSH.")

        # Install dependencies and deploy model
        commands = [
            "sudo yum update -y",
            "sudo yum install docker -y",
            "sudo service docker start",
            f"aws s3 cp s3://{S3_BUCKET_NAME}/{os.path.basename(MODEL_FILE_PATH)} /home/ec2-user/",
            "docker pull sentiment-analysis-server-image",
            f"bash {DEPLOY_SCRIPT_PATH}"
        ]

        for cmd in commands:
            stdin, stdout, stderr = ssh_client.exec_command(cmd)
            logger.info(stdout.read().decode())
            if stderr:
                logger.error(stderr.read().decode())

        logger.info(f"EC2 instance {public_dns} configured successfully.")
    except Exception as e:
        logger.error(f"Error configuring EC2 instance: {e}")
        raise
    finally:
        ssh_client.close()

# Step 6: Terminate EC2 instance
def terminate_ec2_instance(instance_id):
    try:
        logger.info(f"Terminating EC2 instance {instance_id}...")
        ec2.terminate_instances(InstanceIds=[instance_id])

        # Wait for the instance to terminate
        waiter = ec2.get_waiter('instance_terminated')
        waiter.wait(InstanceIds=[instance_id])
        logger.info(f"EC2 instance {instance_id} terminated.")
    except ClientError as e:
        logger.error(f"Error terminating EC2 instance: {e}")
        raise

# Step 7: Main deployment process
def main():
    try:
        upload_model_to_s3()
        security_group_id = create_security_group()
        instance_id, public_dns = launch_ec2_instance(security_group_id)
        configure_ec2_instance(public_dns)
        logger.info(f"Deployment complete. Access the server at {public_dns}.")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
    finally:
        terminate_ec2_instance(instance_id)

if __name__ == "__main__":
    main()