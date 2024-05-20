from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import pandas as pd
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from typing import List
import uvicorn
from cachetools import TTLCache
from pymongo import MongoClient
from fuzzywuzzy import fuzz
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# CORS Configuration
origins = ["*"]  # Update this with your UI's domain(s)

# Add CORS middleware
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

# Dataset path (Specify the path to your dataset file)
#dataset_path = "ccg_survey_continued_amber.csv"
#dataset_path = "test_dataset.csv"

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

# Load data
#df = load_data(dataset_path)

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if OpenAI API key is available
if not openai_api_key:
    print("OpenAI API key is missing. Please make sure it's set in your environment variables.")
    exit()

# Initialize ChatOpenAI instance
llm = ChatOpenAI(
    temperature=0, model="gpt-4-turbo", openai_api_key=openai_api_key, streaming=True
)

# Create agent for pandas DataFrame
pandas_df_agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
)

# # Define model for conversation history 
# class ConversationHistory(BaseModel):
#     user_id: str
#     conversations: List[dict]

# Set the TTL in seconds (1 day = 24 hours * 60 minutes * 60 seconds)
ttl_seconds = 24 * 60 * 60

# Initialize TTLCache for caching responses with a 1-day expiry time
response_cache = TTLCache(maxsize=128, ttl=ttl_seconds)

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
    threshold = 80  # Adjust the threshold as needed
    for keyword in bot_identity_keywords:
        if fuzz.partial_ratio(prompt.lower(), keyword.lower()) >= threshold:
            return True
    return False



#get the questions and file path
@app.get("/questions/{user_id}/{survey_id}")
def get_questions(user_id: str, survey_id: str):
    # Find the document containing the questions
    document = collection.find_one({"user_id": user_id, "survey_id": survey_id})

    if document:
        # Extract the questions and file_path from the document
        questions = document.get("questions", [])
        file_path = document.get("file_path", "")
        
        # Pass the file_path to a different method
        df= load_data(file_path)
        print(df)
        
        return {"questions": questions}
    else:
        return {"message": "Document not found"}

# Route to handle user interactions
@app.post("/chat/{user_id}/{survey_id}")
async def generateText(user_id: str, survey_id: str, request: Request) -> JSONResponse:
    prompt = (await request.json()).get("prompt")
    
   # print("Received prompt:", prompt)  # Debugging statement

    # Check if conversation history exists in MongoDB
    existing_history = collection.find_one({"user_id": user_id, "survey_id": survey_id})
    if existing_history:
        # Extract the 'conversations' field from 'existing_history'
        conversations = existing_history.get("conversations", [])
    else:
        conversations = []

    # Add the current conversation to the history
    conversations.append({"user_content": prompt, "bot_content": ""})

    # Check if the response is cached
    if prompt in response_cache:
        response = response_cache[prompt]
    else:
        if is_similar_to_bot_identity(prompt):
          #  print("Bot identity prompt detected.")  # Debugging statement
            response = "I am CML Copilot, how can I help you?"
        else:
            response = pandas_df_agent.run(conversations)

            # Filter out visualization-related responses
            response = filter_response(response)

            # Cache the response with a 1-day expiry time
            response_cache[prompt] = response

    # Update the last conversation with the bot's response
    conversations[-1]["bot_content"] = response

    # Save the conversation history to MongoDB
    collection.update_one({"user_id": user_id, "survey_id": survey_id}, {"$set": {"conversations": conversations}}, upsert=True)

    ret = {"text": response}  # Only return the bot response
    return JSONResponse(ret)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
