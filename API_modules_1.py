from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import pandas as pd
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from typing import List
import uvicorn
import requests
from cachetools import TTLCache
from pymongo import MongoClient
from fuzzywuzzy import fuzz
from fastapi.middleware.cors import CORSMiddleware
import glob
from download_s3 import download_file
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# CORS Configuration
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# MongoDB connection
mongo_uri = os.getenv("MONGODB_URL")
client = MongoClient(mongo_uri)
db = client["convertml-test"]
collection = db["datatango"]

# Supported file formats for uploading
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}


def extract_project_id(file_path):
    # Extract the project ID from the file name
    filename = os.path.basename(file_path)
    project_id = filename.split("_")[0]
    return project_id

def read_csv_file(user_id, project_id):
    # Construct the directory path
    directory_path = f"/home/ubuntu/Satya/DataTango/tmp/{user_id}/gold/"

    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    # Iterate over each CSV file
    for file_path in csv_files:
        # Extract the project ID from the file name
        file_project_id = extract_project_id(os.path.basename(file_path))

        # Check if the project ID matches the provided project_id
        if file_project_id == project_id:
            # Read the CSV file into a Pandas DataFrame
            print(file_path)
            df = pd.read_csv(file_path)

            return df
    else:
        print("Error: More than one CSV file found or no CSV file found in the directory.")
        return None

# def read_csv_file(userid):
#     # Construct the directory path
#     directory_path = f"/home/ubuntu/Satya/DataTango/tmp/{userid}/gold/"

#     # Find the CSV file in the directory
#     csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

#     if len(csv_files) == 1:
#         # Assuming there is only one CSV file, directly access the first file in the list
#         file_path = csv_files[0]

#         # Read the CSV file into a Pandas DataFrame
#         df = pd.read_csv(file_path)

#         return df
#     else:
#         print("Error: More than one CSV file found or no CSV file found in the directory.")
#         return None

# Function to load data from provided file path
def load_data(file_path):
    try:
        ext = os.path.splitext(file_path)[1][1:].lower()
    except:
        ext = file_path.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](file_path)
    else:
        print(f"Unsupported file format: {ext}")
        return None

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if OpenAI API key is available
if not openai_api_key:
    print("OpenAI API key is missing. Please make sure it's set in your environment variables.")
    exit()

# Initialize TTLCache for caching responses with a 1-day expiry time
response_cache = TTLCache(maxsize=128, ttl=24 * 60 * 60)

# Initialize ChatOpenAI instance
llm = ChatOpenAI(
    temperature=0, model="gpt-4-turbo", openai_api_key=openai_api_key, streaming=True
)

# Initialize pandas DataFrame agents
pandas_df_agents = {}

# Function to filter out visualization-related responses
def filter_response(response):
    visualization_phrases = [
        "don't have access to the necessary tools",
        "create a visualization directly",
        "guide you on how to create the visualization using Python",
        "what kind of visualization you need",
        "what data from your dataframe you would like to visualize",
        "bar chart",
        "histogram",
        "scatter plot",
        "any other type of visual representation",
        "Let me know the details!",
    ]
    matplotlib_installation_phrase = "pip install matplotlib"

    if not any(phrase in response for phrase in visualization_phrases) and matplotlib_installation_phrase not in response:
        return response
    else:
        return "CML Copilot: I don't have any specific visualization to show you right now. Is there anything else I can assist you with?"

# Check if the user input is similar to a bot identity prompt
def is_similar_to_bot_identity(prompt):
    bot_identity_keywords = [
        "who are you",
        "what are you",
        "your name",
        "what's your name",
        "tell me about yourself",
        "introduce yourself",
        "what can you do",
        "what do you do",
        "what can you help with",
        "what is your purpose",
        "are you a bot",
        "are you human",
        "are you real",
        "are you a program",
        "are you a virtual assistant",
        "are you a language model",
        "what do people call you",
        "do you have a nickname",
        "want to know about you",
    ]
    threshold = 80
    for keyword in bot_identity_keywords:
        if fuzz.partial_ratio(prompt.lower(), keyword.lower()) >= threshold:
            return True
    return False


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




@app.get("/questions1/{user_id}/{project_id}")
def get_questions(user_id: str, project_id: str):
    # Find the document containing the questions
    document = collection.find_one({"user_id": user_id, "project_id": project_id})

    if document:
        # Extract the questions from the document
        questions = document.get("questions", [])
    else:
        return {"message": "Document not found"}

    return {"questions": questions}




# Get the questions from MongoDB
@app.get("/questions/{user_id}/{project_id}")
def get_questions(user_id: str, project_id: str):
    # Check if the conversations are already cached
    cache_key = f"conversations-{user_id}-{project_id}"
    if cache_key in response_cache:
        conversations = response_cache[cache_key]["conversations"]
    else:
        # Initialize an empty list if conversations are not cached
        conversations = []

    # Find the document containing the questions
    document = collection.find_one({"user_id": user_id, "project_id": project_id})

    if document:
        # Extract the questions and file_path from the document
        questions = document.get("questions", [])
      
        df=read_csv_file(user_id,project_id)

        if df is None:
            return {"message": "Failed to load DataFrame"}

        # Cache the conversations and DataFrame
        response_cache[cache_key] = {"conversations": conversations, "df": df}
    else:
        return {"message": "Document not found"}

    # Create agent for pandas DataFrame if not already created
    if cache_key not in pandas_df_agents:
        pandas_df_agents[cache_key] = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
        )

    return {"questions": questions}


# Route to handle user interactions
@app.post("/chat/{user_id}/{project_id}")
async def generateText(user_id: str, project_id: str, request: Request) -> JSONResponse:
    prompt = (await request.json()).get("prompt")

    # Check if the conversations are already cached
    cache_key = f"conversations-{user_id}-{project_id}"
    if cache_key in response_cache:
        conversations = response_cache[cache_key]["conversations"]
    else:
        # Initialize an empty list if conversations are not cached
        conversations = []

    # Check if the response is cached based on conversation history
    conversation_key = "-".join([conv["user_content"] for conv in conversations])
    cache_key = f"responses-{user_id}-{project_id}-{conversation_key}"
    if cache_key in response_cache:
        response = response_cache[cache_key]
    else:
        if is_similar_to_bot_identity(prompt):
            response = "I am CML Copilot, how can I help you?"
        else:
            # Check if the pandas DataFrame agent is initialized for the cache_key
            if cache_key not in pandas_df_agents:
                # Find the document containing the questions
                document = collection.find_one({"user_id": user_id, "project_id": project_id})

                if document:
                    # Extract the questions and file_path from the document
                    df=read_csv_file(user_id,project_id)

                    if df is None:
                        return JSONResponse({"text": "Failed to load DataFrame"})

                    pandas_df_agents[cache_key] = create_pandas_dataframe_agent(
                        llm,
                        df,
                        verbose=True,
                        agent_type=AgentType.OPENAI_FUNCTIONS,
                        handle_parsing_errors=True,
                    )
                else:
                    return JSONResponse({"text": "Document not found"})

            response = pandas_df_agents[cache_key].run(conversations)

            # Filter out visualization-related responses
            response = filter_response(response)

            # Cache the response with a 1-day expiry time
            response_cache[cache_key] = response

    # Update the conversation history with the bot's response
    conversations.append({"user_content": prompt, "bot_content": response})

    # Update the cache with the updated conversations
    response_cache[f"conversations-{user_id}-{project_id}"] = {"conversations": conversations}

    # Update the conversation history in MongoDB
    collection.update_one(
        {"user_id": user_id, "project_id": project_id},
        {"$set": {"conversations": conversations}},
        upsert=True
    )

    return JSONResponse({"text": response})

# Route to handle user interactions
@app.post("/chat1/{user_id}/{project_id}")
async def generateText(user_id: str, project_id: str, request: Request) -> JSONResponse:
    prompt = (await request.json()).get("prompt")

    # Check if the conversations are already cached
    cache_key = f"conversations-{user_id}-{project_id}"
    if cache_key in response_cache:
        conversations = response_cache[cache_key]["conversations"]
    else:
        # Initialize an empty list if conversations are not cached
        conversations = []

    # Add the current conversation to the history
    conversations.append({"user_content": prompt, "bot_content": ""})

    # Check if the response is cached based on conversation history
    conversation_key = "-".join([conv["user_content"] for conv in conversations])
    cache_key = f"responses-{user_id}-{project_id}-{conversation_key}"
    if cache_key in response_cache:
        response = response_cache[cache_key]
    else:
        if is_similar_to_bot_identity(prompt):
            response = "I am CML Copilot, how can I help you?"
        else:
            # Check if the pandas DataFrame agent is initialized for the cache_key
            if cache_key not in pandas_df_agents:
                # Find the document containing the questions
                document = collection.find_one({"user_id": user_id, "project_id": project_id})

                if document:
                    # Extract the questions and file_path from the document
                    questions = document.get("questions", [])
                    file_path = document.get("file_name", "")

                    # Load the dataset using the file_path
                    #df = load_data(file_path)   ### as per userid-----------------------------------------------------
                    df=read_csv_file(user_id,project_id)

                    if df is None:
                        return JSONResponse({"text": "Failed to load DataFrame"})

                    pandas_df_agents[cache_key] = create_pandas_dataframe_agent(
                        llm,
                        df,
                        verbose=True,
                        agent_type=AgentType.OPENAI_FUNCTIONS,
                        handle_parsing_errors=True,
                    )
                else:
                    return JSONResponse({"text": "Document not found"})

            response = pandas_df_agents[cache_key].run(conversations)

            # Filter out visualization-related responses
            response = filter_response(response)

            # Cache the response with a 1-day expiry time
            response_cache[cache_key] = response

    # Update the cache with the updated conversations
    response_cache[cache_key] = response
    response_cache[f"conversations-{user_id}-{project_id}"] = {"conversations": conversations}

    return JSONResponse({"text": response})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
