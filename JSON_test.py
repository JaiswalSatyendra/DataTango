import os
import pandas as pd
import glob
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from pymongo import MongoClient
import json


def process_data_and_save_to_mongodb(user_id, project_id):
    # Load the API key and MongoDB URL from the .env file
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    mongo_uri = os.getenv("MONGODB_URL")

    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client["convertml-test"]
    collection = db["datatango"]

        
    df = pd.read_csv("test_dataset.csv")

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
    
# Function to get questions from MongoDB
def get_questions(user_id, survey_id):
    # Load environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    mongo_uri = os.getenv("MONGODB_URL")

    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client["convertml-test"]
    collection = db["datatango"]

    # Find the document containing the questions
    document = collection.find_one({"user_id": user_id, "survey_id": survey_id})

    if document:
        # Extract the questions from the document
        questions = document.get("questions", [])
        return questions
    else:
        return None
# Example usage
if __name__ == "__main__":
    load_dotenv()
    user_id = "12345678"  # Replace with the actual user_id
    project_id = "123"  # Replace with the actual project_id
    survey_id = "survey123" 
   # process_data_and_save_to_mongodb(user_id, project_id)
    questions = get_questions(user_id, survey_id)

    if questions is not None:
        print(f"Questions for user_id {user_id} and survey_id {survey_id}:")
        for question in questions:
            print(question)
    else:
        print(f"No questions found for user_id {user_id} and survey_id {survey_id}.")
