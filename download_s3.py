import ast
from fastapi import FastAPI, Request
import boto3
import os
import json
import requests
import logging
from dotenv import load_dotenv
from read_csv_n_generate import process_data_and_save_to_mongodb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

app = FastAPI()


@app.post("/sns")
async def receive_sns_message(request: Request):
    message = await request.json()
    logger.info("-------")
    logger.info(message)
    header = request.headers["x-amz-sns-message-type"]

    if header == "SubscriptionConfirmation" and "SubscribeURL" in message:
        # Visit the SubscribeURL to confirm the subscription automatically
        subscribe_url = message["SubscribeURL"]
        response = requests.get(subscribe_url)
        return {"status": "Subscription confirmed"}
    elif header == "Notification":
        # Handle notification here
        await download_file(message["Message"])
        return {"status": "file downloaded", "message": message["Message"]}
    else:
        return {"status": "Unsupported SNS message type"}

    return {"status": "OK"}


async def download_file(message):

    bucket_name = message.split("/")[0]
    user_id = message.split("/")[1]
    file_name = message.split("/")[-1]
    s3_key = message.split("/", 1)[1]
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="us-east-1",
    )
    # Create the directory if it doesn't exist
    local_path = f"./tmp/{user_id}/gold/"
    os.makedirs(local_path, exist_ok=True)

    local_path = os.path.join(local_path, file_name)
    s3.download_file(bucket_name, s3_key, local_path)

    # get he project_id
    parts = s3_key.split('/')
    project_id = parts[-1].split('_')[0]
    process_data_and_save_to_mongodb(user_id,project_id)


if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)
