import os
import pandas as pd
import glob
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from pymongo import MongoClient
import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_project_id(file_path):
    # Extract the project ID from the file name
    filename = os.path.basename(file_path)
    project_id = filename.split("_")[0]
    return project_id

def process_data_and_save_to_mongodb(user_id, project_id):
    # Load the API key and MongoDB URL from the .env file
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    mongo_uri = os.getenv("MONGODB_URL")

    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client["convertml-test"]
    collection = db["datatango"]

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
            prompt = f"Given the survey data, please provide six key questions without explanations only questions that would be beneficial for a customer to understand their consumers. Focus on demographics, preferences, and feedback. Highlight trends, patterns, and outliers."

            # Use the agent to generate questions based on the prompt
            messages = [{"role": "user", "content": prompt}]
            response = pandas_df_agent.run(messages)

            # Split the response into individual questions
            questions = response.strip().split("\n")

            # Prepare the document to insert into MongoDB
            document = {
                "user_id": user_id,
                "project_id": project_id,
                "questions": [question.strip() for question in questions]
            }

            # Insert the document into MongoDB
            collection.insert_one(document)

            # Exit the loop as we have found and processed the matching file
            break

    else:
        logger.info("No matching CSV file found in the directory.")

# Example usage
# if __name__ == "__main__":
#     user_id = "65d5e841af6e64f8c032e8ed"  # Replace with the actual user_id
#     project_id = "6605470be7c1463e26b8fe29"  # Replace with the actual project_id
#     process_data_and_save_to_mongodb(user_id, project_id)
