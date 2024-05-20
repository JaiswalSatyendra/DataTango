import os
from dotenv import load_dotenv
import glob
import pandas as pd
import openai
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

    if len(csv_files) == 1:
        # Assuming there is only one CSV file, directly access the first file in the list
        file_path = csv_files[0]

        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file_path)

        # Define the llm variable
        llm = ChatOpenAI(
            temperature=0, model="gpt-4-turbo", openai_api_key=openai_api_key, streaming=True
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
        prompt = f"Given the survey data for survey, please provide six key questions that would be beneficial for a customer to understand their consumers. The dataset includes information such as demographics, preferences, and feedback. Focus on trends, patterns, and outliers that could help improve customer understanding and decision-making."

        # Use the agent to generate questions based on the prompt
        messages = [{"role": "user", "content": prompt}]
        response = pandas_df_agent.run(messages)

        # Split the response into individual questions
        questions = response.strip().split("\n\n")

        # Create a dictionary to store the questions
        questions_dict = {}
        for q in questions:
            q_parts = q.split("\n")
            q_num = int(q_parts[0].split(".")[0])
            q_text = q_parts[1].split(":")[1].strip()
            questions_dict[q_num] = {"question": q_text}

        # Convert the dictionary to JSON format
        questions_json = json.dumps(questions_dict, indent=4)

        print(questions_json)

        # Prepare the document to insert into MongoDB
        document = {
            "user_id": user_id,
            "survey_id": survey_id,
            "file_name": file_path.split(os.path.sep)[-1],
            "questions": questions_json,
        }

        # Insert the document into MongoDB
        collection.insert_one(document)

    

        # Return the questions as a JSON object
        return json.loads(response)

    else:
        print("No CSV file found in the directory or more than one CSV file found. Please ensure there is exactly one CSV file.")

# Usage
if __name__ == "__main__":
    user_id = "123456789"  # Replace with the actual user_id
    survey_id = "survey123"  # Replace with the actual survey_id
    questions_json = process_data_and_save_to_mongodb(user_id, survey_id)
