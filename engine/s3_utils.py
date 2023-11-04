# s3_utils.py
import boto3
from botocore.exceptions import ClientError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def create_bucket(bucket_name, region=None):
    """Create an S3 bucket in a specified region"""
    try:
        if region is None:
            s3_client = boto3.client('s3')
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client('s3', region_name=region)
            location = {'LocationConstraint': region}
            s3_client.create_bucket(
                Bucket=bucket_name, CreateBucketConfiguration=location)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def bucket_exists(bucket_name):
    """Check if an S3 bucket exists"""
    s3 = boto3.client('s3')
    try:
        s3.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        logging.error(e)
        return False


def upload_file_to_s3(local_file_path, bucket_name, s3_file_name):
    """Upload a file to an S3 bucket"""
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_file_name)
        return True
    except ClientError as e:
        logging.error(e)
        return False


def list_files_in_bucket(bucket_name):
    """List files in an S3 bucket"""
    s3 = boto3.client('s3')
    try:
        contents = []
        for item in s3.list_objects(Bucket=bucket_name)['Contents']:
            contents.append(item['Key'])
        return contents
    except ClientError as e:
        logging.error(e)
        return None
