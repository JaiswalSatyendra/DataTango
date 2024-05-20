import streamlit as st
import pandas as pd
import os
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

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

# Function to clear submit button state
def clear_submit():
    st.session_state["submit"] = False

# Function to load data from uploaded file
@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

# Set Streamlit page configuration
st.set_page_config(page_title="LangChain: Chat with pandas DataFrame", page_icon=" :) ")
st.title("Chat with your DATA")

# File upload section
uploaded_file = st.file_uploader(
    "Upload a Data file",
    type=list(file_formats.keys()),
    help="Various File formats are Support",
    on_change=clear_submit,
)

# Warning message for security
if not uploaded_file:
    st.warning(
        "This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
    )

# Load data if file is uploaded
if uploaded_file:
    df = load_data(uploaded_file)

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if OpenAI API key is available
if not openai_api_key:
    st.error("OpenAI API key is missing. Please make sure it's set in your environment variables.")
    st.stop()

# Clear conversation history button
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "CML Copilot", "content": "How can I help you?"}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input prompt from user
if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

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
        st.session_state.messages.append({"role": "CML Copilot", "content": "CML Copilot"})
        st.write("CML Copilot")  # Display the bot's identity
        
    else:
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

        # Execute agent and display response
        with st.chat_message("CML Copilot"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])

            # Filter out visualization-related responses
            visualization_phrases = [
                "don't have access to the necessary tools",
                "create a visualization directly",
                "guide you on how to create the visualization using Python"
            ]

            # Filter out Matplotlib installation response
            matplotlib_installation_phrase = "pip install matplotlib"

            if not any(phrase in response for phrase in visualization_phrases) and matplotlib_installation_phrase not in response:
                st.session_state.messages.append({"role": "CML Copilot", "content": response})
                st.write(response)  # Display the filtered response
            else:
                st.session_state.messages.append({"role": "CML Copilot", "content": response})
                st.write("CML Copilot: I don't have any specific visualization to show you right now. Is there anything else I can assist you with?")
