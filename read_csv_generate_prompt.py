import os
from dotenv import load_dotenv
import glob
import pandas as pd
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from pymongo import MongoClient
import json

def process_data_and_save_to_mongodb(user_id, survey_id):
    # Load the API key and MongoDB URL from the .env file
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    mongo_uri = os.getenv("MONGODB_URL")

    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client["convertml-test"]
    collection = db["datatango"]

    # Define the directory path
    directory_path = r'C:\Users\SatyendraJaiswal\Desktop\ConvertML\ChatBot\RAG\agent\CSV-DATA'

    # Find the CSV file in the directory
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    if len(csv_files) >= 1:
        # Assuming there is only one CSV file, directly access the first file in the list
        file_path = csv_files[0]

        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file_path)

        # Define the llm variable
        llm = ChatOpenAI(
            temperature=0, model="gpt-4o-2024-05-13", openai_api_key=openai_api_key, streaming=True
        )

        # Create a Pandas DataFrame agent
        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
        )

        # Define the prompt
        prompt = f"Given the survey data, please provide six key questions without explanations only questions that would be beneficial for a customer to understand their consumers. Focus on demographics, preferences, and feedback. Highlight trends, patterns, and outliers."

        # Use the agent to generate questions based on the prompt
        messages = [{"role": "user", "content": prompt}]
        response = pandas_df_agent.run(messages)
        # print(type(response))

        # Split the response into individual questions
        # Split the response into individual questions
        questions = response.strip().split("\n")


        # Prepare the document to insert into MongoDB
        document = {
            "user_id": user_id,
            "survey_id": survey_id,
            "file_name": file_path.split(os.path.sep)[-1],
            "questions": [question.strip() for question in questions]
        }

        # Insert the document into MongoDB
        collection.insert_one(document)

    else:
        print("No CSV file found in the directory or more than one CSV file found. Please ensure there is exactly one CSV file.")

# Usage
if __name__ == "__main__":
    user_id = "12345678"  # Replace with the actual user_id
    survey_id = "survey123"  # Replace with the actual survey_id
    process_data_and_save_to_mongodb(user_id, survey_id)
