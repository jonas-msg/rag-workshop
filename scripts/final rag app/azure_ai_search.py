import os
from dotenv import load_dotenv
from openai import AzureOpenAI

from langchain_openai import AzureChatOpenAI
from azure.core.credentials import AzureKeyCredential
from langchain.chains import LLMChain, ConversationChain, RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.memory import  ConversationBufferMemory
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from azure.search.documents import SearchClient
from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.vectorstores import chroma
import streamlit as st
from langchain.vectorstores import AzureSearch

load_dotenv()

#importing Azure OpenAI creds
api_key = os.getenv("AZURE_OPENAI_KEY")
os.environ["OPENAI_API_KEY"]= api_key
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name='gpt' 
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

#initiate AI Searchservice
service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
cogkey = os.getenv("AZURE_SEARCH_ADMIN_KEY")
credential = AzureKeyCredential(key)
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002", chunk_size=1)

# Set our Azure Search
acs = AzureSearch(azure_search_endpoint=os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT'),
                 azure_search_key=os.getenv('AZURE_SEARCH_ADMIN_KEY'),
                 index_name=index_name,
                 embedding_function=embeddings.embed_query)

#create retriever: retrieving relevant documents for query
retriever = AzureCognitiveSearchRetriever(service_name="jonassearch",api_key=cogkey,index_name=index_name, content_key="line", top_k=3)

def get_doc_azure_ai(prompt):
    return [doc.page_content for doc in retriever.get_relevant_documents(prompt)]