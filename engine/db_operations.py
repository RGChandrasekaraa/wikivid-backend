import time
from datetime import datetime
from botocore.exceptions import ClientError
import boto3

# Set up DynamoDB
dynamodb = boto3.resource('dynamodb', region_name='ap-southeast-2')
table = dynamodb.Table('A2VRequestsDB')


def create_request_entry(token, url):
    try:
        current_time_formatted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        response = table.put_item(
            Item={
                'Token': token,
                'URL': url,
                'Status': 'INITIATED',
                'RequestedAt': current_time_formatted,
                'VideoURL': None,
            }
        )
        return response
    except ClientError as e:
        print(f"Failed to create entry in DynamoDB: {e}")
        return None


def update_request_status(token, status, video_url=None):
    try:
        updated_at_formatted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        expression_attribute_values = {
            ':status': status,
            ':completed_at': updated_at_formatted
        }

        update_expression = 'SET #status_attr = :status, CompletedAt = :completed_at'

        if video_url:
            update_expression += ', VideoURL = :video_url'
            expression_attribute_values[':video_url'] = video_url

        response = table.update_item(
            Key={'Token': token},
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_attribute_values,
            ExpressionAttributeNames={
                '#status_attr': 'Status'  # Placeholder for the reserved attribute name
            }
        )
        return response
    except ClientError as e:
        print(f"Failed to update entry in DynamoDB: {e}")
        return None


def finalize_request(token, video_url):
    # Now it just updates the status and video URL without a completion time
    return update_request_status(token, 'Completed', video_url=video_url)


def get_status_by_token(token):
    try:
        response = table.get_item(Key={'Token': token})
        if 'Item' in response:
            return response['Item']
        else:
            print(f"No item found with token: {token}")
            return None
    except ClientError as e:
        print(f"Failed to get entry in DynamoDB: {e}")
        return None
