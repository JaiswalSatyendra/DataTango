import pandas as pd
import os
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
#from langchain.callbacks import CallbackHandler
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

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

# Function to connect to MongoDB
def connect_to_mongodb():
    client = MongoClient('mongodb+srv://convertml:0uZHJpgF3LVwbESj@convertml.yximg.mongodb.net/convertml?retryWrites=true&w=majority')  # Assuming MongoDB is running on localhost
    return client

# Function to load conversation history from MongoDB
def load_conversation_history(collection):
    return list(collection.find())

# Function to save conversation history to MongoDB
def save_conversation_history(collection, conversation_history):
    collection.insert_many(conversation_history)

# Load data
def main():
    # Load data from the specified dataset path
    df = load_data(dataset_path)

    # Retrieve OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Check if OpenAI API key is available
    if not openai_api_key:
        print("OpenAI API key is missing. Please make sure it's set in your environment variables.")
        return

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

    # Connect to MongoDB
    client = connect_to_mongodb()
    db = client["convertml-test"]  # Use the "convertml-test" database
    collection = db["datatango"]  # Use the "datatango" collection

    # Load conversation history from MongoDB
    conversation_history = load_conversation_history(collection)

    # Chat loop
    while True:
        # Input prompt from user
        prompt = input("You: ")

        messages = [{"role": "user", "content": prompt}]

        # Execute agent and display response
        response = pandas_df_agent.run(messages)

        # Filter out visualization-related responses
        visualization_phrases = [
            "don't have access to the necessary tools",
            "create a visualization directly",
            "guide you on how to create the visualization using Python"
        ]

        # Filter out Matplotlib installation response
        matplotlib_installation_phrase = "pip install matplotlib"

        if not any(phrase in response for phrase in visualization_phrases) and matplotlib_installation_phrase not in response:
            print("Bot:", response)  # Display the response
        else:
            # Display a customized message if the response is related to visualization or Matplotlib installation
            print("Bot: I don't have any specific visualization to show you right now. Is there anything else I can assist you with?")

        # Store user input and bot response in conversation history
        conversation_history.append({"role": "user", "content": prompt})
        conversation_history.append({"role": "bot", "content": response})

        # Save conversation history to MongoDB
        save_conversation_history(collection, conversation_history)


if __name__ == "__main__":
    main()
