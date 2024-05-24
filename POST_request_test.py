from fastapi import FastAPI, HTTPException, Request
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
from langchain_core.exceptions import OutputParserException
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







# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if OpenAI API key is available
if not openai_api_key:
    print("OpenAI API key is missing. Please make sure it's set in your environment variables.")
    exit()

# Initialize TTLCache for caching responses with a 1-day expiry time
response_cache = TTLCache(maxsize=128, ttl=24 * 60 * 60)

 #temperature=0, model="gpt-4o-2024-05-13", openai_api_key=openai_api_key, streaming=True

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
        "hello, who are you?"
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



# Unwanted keywords or phrases indicating irrelevant responses
unwanted_keywords = [
    "data preparation", "data cleaning", "exploratory data analysis", 
    "statistical analysis", "regression model", "boxplot", "scatter plot"
]

def contains_unwanted_keywords(response: str, keywords: list) -> bool:
    return any(keyword in response.lower() for keyword in keywords)



def read_csv_file(user_id, project_id):
    # Construct the directory path
    
    df = pd.read_csv("Airlines.csv")
    print(df.head(2))
    return df




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
       # file_path = document.get("file_name", "")

        # Load the dataset using the file_path
       # df = load_data(file_path)   ### as per userid-----------------------------------------------------
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
@app.post("/chat1/{user_id}/{project_id}")
async def generateText(user_id: str, project_id: str, request: Request) -> JSONResponse:
    prompt = (await request.json()).get("prompt")


    # Find the document containing the questions
    document = collection.find_one({"user_id": user_id, "project_id": project_id})
    if document:
        conversations = document.get("conversations", [])
    else:
        conversations = []
    # Add the current conversation to the history
    conversations.append({"user_content": prompt, "bot_content": ""})    


    # Check if the pandas DataFrame agent is initialized for the cache_key
    cache_key = f"pandas_df_agent-{user_id}-{project_id}"

    if is_similar_to_bot_identity(prompt):
        response = "I am CML Copilot, how can I help you?"

    else:
        if cache_key not in pandas_df_agents:
        # Extract the questions and file_path from the document
            df = read_csv_file(user_id, project_id)

          #  df=pd.read_csv("test_dataset.csv")
            if df is None:
                return JSONResponse({"text": "Failed to load DataFrame"})

            pandas_df_agents[cache_key] = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True,
            )

        # Get the latest conversation from the cache
        latest_conversation = conversations[-1]["user_content"]

        # Generate the response using the pandas DataFrame agent
        response = pandas_df_agents[cache_key].run(latest_conversation)

        # Filter out visualization-related responses
        response = filter_response(response)
    # Update the last conversation with the bot's response
    conversations[-1]["bot_content"] = response
   # print("final data-->",conversations)
    # Update the conversation history in MongoDB
 #   collection.update_one({"user_id": user_id, "survey_id": project_id}, {"$set": {"conversations": conversations}}, upsert=True)

    collection.update_one(
        {"user_id": user_id, "project_id": project_id},
        {"$set": {"conversations": conversations}},
        upsert=True
    )
    x=type(response)
    print("----------",x)
    print(response)
    #return JSONResponse(response)
    return response
    #return JSONResponse({"text": response})

@app.post("/chat/{user_id}/{project_id}")
async def generate_text(user_id: str, project_id: str, request: Request) -> JSONResponse:
    prompt = (await request.json()).get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Validate user_id and project_id inputs
    # Add your validation logic here

    # Find the document containing the questions
    document = collection.find_one({"user_id": user_id, "project_id": project_id})
    if document:
        conversations = document.get("conversations", [])
    else:
        conversations = []

    # Add the current conversation to the history
    conversations.append({"user_content": prompt, "bot_content": ""})

    # Check if the pandas DataFrame agent is initialized for the cache_key
    cache_key = f"pandas_df_agent-{user_id}-{project_id}"

    if is_similar_to_bot_identity(prompt):
        response = "I am CML Copilot, how can I help you?"
    else:
        if cache_key not in pandas_df_agents:
            df = read_csv_file(user_id, project_id)
            if df is None:
                response = "Failed to load DataFrame"
            else:
                pandas_df_agents[cache_key] = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    handle_parsing_errors=True,
                )
                response = pandas_df_agents[cache_key].run(prompt)
        else:
            # Get the latest conversation from the cache
            latest_conversation = conversations[-1]["user_content"]

            # Generate the response using the pandas DataFrame agent
            response = pandas_df_agents[cache_key].run(latest_conversation)

        # Filter out unwanted responses
        if contains_unwanted_keywords(response, unwanted_keywords):
            response = "I am not able to assist with detailed data analysis steps. Please ask a specific question related to the data."

        # Filter out visualization-related responses
        response = filter_response(response)

    # Update the last conversation with the bot's response
    conversations[-1]["bot_content"] = response

    # Update the conversation history in MongoDB
    collection.update_one(
        {"user_id": user_id, "project_id": project_id},
        {"$set": {"conversations": conversations}},
        upsert=True
    )

    # Ensure the response is returned as JSON
    print(type(response))
    print("------------------------------------------")
    print(response)
    return JSONResponse({"text": response})



@app.post("/chatx/{user_id}/{project_id}")
async def generate_text(user_id: str, project_id: str, request: Request) -> JSONResponse:
    prompt = (await request.json()).get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Validate user_id and project_id inputs
    # Add your validation logic here

    # Find the document containing the questions
    document = collection.find_one({"user_id": user_id, "project_id": project_id})
    if document:
        conversations = document.get("conversations", [])
    else:
        conversations = []

    # Add the current conversation to the history
    conversations.append({"user_content": prompt, "bot_content": ""})

    # Check if the pandas DataFrame agent is initialized for the cache_key
    cache_key = f"pandas_df_agent-{user_id}-{project_id}"

    if is_similar_to_bot_identity(prompt):
        response = "I am CML Copilot, how can I help you?"
    else:
        if cache_key not in pandas_df_agents:
            df = read_csv_file(user_id, project_id)
            if df is None:
                response = "Failed to load DataFrame"
            else:
                pandas_df_agents[cache_key] = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                   # agent_type="OPENAI_FUNCTIONS",
                    agent_type="openai-tools",
                    handle_parsing_errors=True,
                )
                try:
                    response = pandas_df_agents[cache_key].run(prompt)
                except OutputParserException as e:
                    response = str(e)
        else:
            # Generate the response using the pandas DataFrame agent
            try:
                response = pandas_df_agents[cache_key].run(prompt)
            except OutputParserException as e:
                response = str(e)

        # Filter out unwanted responses
        if contains_unwanted_keywords(response, unwanted_keywords):
            response = "I am not able to assist with detailed data analysis steps. Please ask a specific question related to the data."

        # Filter out visualization-related responses
        response = filter_response(response)

    # Update the last conversation with the bot's response
    conversations[-1]["bot_content"] = response

    # Update the conversation history in MongoDB
    collection.update_one(
        {"user_id": user_id, "project_id": project_id},
        {"$set": {"conversations": conversations}},
        upsert=True
    )

    # Ensure the response is returned as JSON
    return JSONResponse({"text": response})    



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
