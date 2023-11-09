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


def upload_fileobj_to_s3(file_obj, bucket_name, object_name, acl='private'):
    """
    Upload a file object to an S3 bucket

    :param file_obj: File object to upload.
    :param bucket_name: Bucket to upload to.
    :param object_name: S3 object name. If not specified then file_obj name is used.
    :param acl: String. The canned ACL to apply to the object.
    :return: True if file was uploaded, else False.
    """
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_fileobj(
            file_obj,
            bucket_name,
            object_name,
            ExtraArgs={'ACL': acl}
        )
        logging.info(f"File {object_name} uploaded to {bucket_name}")
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
