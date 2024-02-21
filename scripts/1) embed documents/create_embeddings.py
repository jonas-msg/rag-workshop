from dotenv import load_dotenv, dotenv_values
import os
from openai import AzureOpenAI
import re
import json
import glob, os
from PyPDF2 import PdfReader

load_dotenv()
api_key = os.getenv("AZURE_OPENAI_KEY")
api_version = "2023-12-01-preview"
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
engine = "text-embedding-ada-002"

client = AzureOpenAI(
    api_key=api_key, api_version="2023-12-01-preview", azure_endpoint=azure_endpoint
)


# initializing langchain with AzureOpenAI as LLM (large language model)
# llm = AzureChatOpenAI(
#     openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
#     api_version="2023-12-01-preview",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     deployment_name=os.getenv("DEPLOYMENT_NAME"),
#     model_name="gpt-35-turbo",
#     temperature=0.9,
# )


def get_files_from_data_dir():
    cwd = os.getcwd()
    relative_path = "../data"
    absolute_path = os.path.abspath(os.path.join(cwd, relative_path))
    print(f"absolute_path: {absolute_path}")

    pdfs = []
    for file in glob.glob("../../data/*.pdf"):
        print(f"file{file}")
        filepath = os.path.abspath(os.path.join(absolute_path, file))
        pdfs.append(filepath)
    return pdfs


def readfiles(file_path):
    os.chdir(file_path)
    pdfs = []
    for file in glob.glob("*.pdf"):
        pdfs.append(file)
    return pdfs


# Read the PDF file and return the text
def get_pdf_data(file_path, num_pages=1):
    reader = PdfReader(file_path)
    full_doc_text = ""
    pages = reader.pages
    num_pages = len(pages)

    try:
        for page in range(num_pages):
            current_page = reader.pages[page]
            text = current_page.extract_text()
            full_doc_text += text
    except:
        print("Error reading file")
    finally:
        return full_doc_text


def normalize_text(string):
    """Normalize text by removing unnecessary characters and altering the format of words."""
    # make text lowercase
    string = re.sub(r"\s+", " ", string).strip()
    string = re.sub(r". ,", "", string)
    # remove all instances of multiple spaces
    string = string.replace("..", ".")
    string = string.replace(". .", ".")
    string = string.replace("\n", "")
    string = string.strip()
    return string


# Divide the text into chunks of chunk_length
# [default is 500] characters
def get_chunks(fulltext: str, chunk_length=500) -> list:
    text = fulltext

    chunks = []
    while len(text) > chunk_length:
        last_period_index = text[:chunk_length].rfind(".")
        if last_period_index == -1:
            last_period_index = chunk_length
        chunks.append(text[:last_period_index])
        text = text[last_period_index + 1 :]
    chunks.append(text)

    return chunks


def get_embedding(engine, text, filename):
    doc_embed = []
    counter = 0
    for line in text:
        d = {}
        d["id"] = str(counter)
        d["line"] = line
        response = client.embeddings.create(input=line, model=engine)
        d["embedding"] = response.data[0].embedding
        d["filename"] = filename
        counter = counter + 1
        doc_embed.append(d)
    return doc_embed


files = get_files_from_data_dir()
print(files)

chunked_files = []
for file in files:
    raw_data = get_pdf_data(file, num_pages=1)
    normalized_data = normalize_text(raw_data)
    chunks = get_chunks(normalized_data, 500)
    chunked_files.append(chunks)


# create embeddings and write to ../../output
for index, chunks in enumerate(chunked_files):
    filename = files[index].split("\\")[-1]
    embeddings_entire_text = get_embedding(engine, chunks, filename)
    with open(f"../output/{filename}.json", "w+", encoding="utf-8") as outfile:
        json.dump(embeddings_entire_text, outfile)
