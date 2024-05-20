import os
from dotenv import load_dotenv
import glob
import pandas as pd
import openai
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI

# Load the API key from the .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

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
    prompt = f"List the top 6 questions from this dataset"

    # Use the agent to generate questions based on the prompt
    messages = [{"role": "user", "content": prompt}]
    response = pandas_df_agent.run(messages)

    # Display the response
    print(response)

else:
    print("No CSV file found in the directory or more than one CSV file found. Please ensure there is exactly one CSV file.")
