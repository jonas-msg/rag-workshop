import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import  ChatMessageHistory, ConversationBufferWindowMemory
import streamlit as st

from azure_ai_search import get_doc_azure_ai

load_dotenv()

#importing Azure OpenAI creds
api_key = os.getenv("AZURE_OPENAI_KEY")
os.environ["OPENAI_API_KEY"]= api_key
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name='gpt' 

# Load environment variables
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Initialize the LLM (Language Model) with Azure OpenAI credentials
llm = AzureChatOpenAI(
    api_version="2023-12-01-preview",
    api_key=api_key,  
    azure_endpoint = azure_endpoint,
    deployment_name="gpt", 
    model_name="gpt-35-turbo",
    temperature=0.9,
    )

# Streamlit UI
st.title("Let's chat")

# Session state to store chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

user_input = st.chat_input("You:", key="input")

# if st.button("Send"):
if user_input:
    # Add user input to history
    st.session_state.history.append(f"Human: {user_input}")
    # st.text(f"You: {user_input}")
    
    # Send the input to the model and get the response
    # response = llm.generate(user_input, max_tokens=100)
    conversation = ConversationChain(llm=llm, verbose=True)
    context = "\n".join(get_doc_azure_ai(user_input))
    messages = "\n\n".join(st.session_state.history)

    rag_messages = f"context: {context}\n\n{messages}"
    print(rag_messages)

    response = conversation.predict(input=messages)
    st.session_state.history.append(f"AI: {response}")
    # st.text(f"ChatGPT: {response}")

# Display chat history
for message in st.session_state.history:
    st.text(message)
