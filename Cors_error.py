from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
import os
import pandas as pd
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from typing import List
import uvicorn

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

# Supported file formats for uploading
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

# Dataset path (Specify the path to your dataset file)
dataset_path = "ccg_survey_continued_amber.csv"

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
df = load_data(dataset_path)

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

# Define model for conversation history
class ConversationHistory(BaseModel):
    user_id: str
    history: List[dict]

# Dictionary to store conversation history for each user session
session_conversations = {}

# Function to filter out visualization-related responses
def filter_response(response):
    visualization_phrases = [
        "don't have access to the necessary tools",
        "create a visualization directly",
        "guide you on how to create the visualization using Python"
    ]
    matplotlib_installation_phrase = "pip install matplotlib"

    if not any(phrase in response for phrase in visualization_phrases) and matplotlib_installation_phrase not in response:
        return response
    else:
        return "CML Copilot: I don't have any specific visualization to show you right now. Is there anything else I can assist you with?"

# Route to handle user interactions
@app.post("/v1/generateText/{user_id}")
async def generateText(user_id: str, request: Request) -> Response:
    prompt = (await request.json()).get("prompt")

    # Get or create conversation history for the user session
    if user_id not in session_conversations:
        session_conversations[user_id] = ConversationHistory(user_id=user_id, history=[])

    conversation_history = session_conversations[user_id].history

    # Add the previous conversation history to the current prompt
    messages = [{"role": "user", "content": prompt}] + conversation_history

    # Check if the user input is a question about the bot's identity
    bot_identity_questions = [
        "who are you",
        "what are you",
        "what's your name",
        "your name",
        "identify yourself",
        "tell me about yourself",
        "introduce yourself",
        "what can you do",
        "what is your purpose",
        "what do you do",
        "what can you help with",
        "are you a bot",
        "what kind of bot are you",
        "are you human or a machine",
        "are you a human or a robot",
        "are you real or artificial",
        "are you a program",
        "are you a virtual assistant",
        "are you a language model",
        "what do people call you",
        "do you have a nickname",
    ]

    if any(question in prompt.lower() for question in bot_identity_questions):
        response = "CML Copilot"
    else:
        response = pandas_df_agent.run(messages)

    # Filter out visualization-related responses
    response = filter_response(response)

    # Store user input and bot response in conversation history
    session_conversations[user_id].history.append({"role": "user", "content": prompt})
    session_conversations[user_id].history.append({"role": "bot", "content": response})

    ret = {"text": response}  # Only return the bot response
    return JSONResponse(ret)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
