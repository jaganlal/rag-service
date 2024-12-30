import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

import preprocessing_service
import chat_service

import openai
from langchain import LangChainClient

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
  raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
langchain_project = os.getenv("LANGCHAIN_PROJECT")
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

openai.api_key = openai_api_key
client = LangChainClient(api_key=langchain_api_key)

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

app = FastAPI(
  title="RAG service",
    description="This service can be used for RAG applications",
    summary="This service can be used for RAG applications",
    version="0.0.1",
    terms_of_service="https://github.com/jaganlal",
    contact={
        "name": "Jaganlal Thoppe",
        "url": "https://github.com/jaganlal/"
    }
)

# Add CORS headers
app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=['*'],
                   allow_headers=['*']
                  )

app.include_router(preprocessing_service.router)
app.include_router(chat_service.router)

# Healthcheck endpoint
@app.get("/api/1.0/ping")
def ping():
    """
    Healthcheck endpoint to verify the service's availability.
    Returns:
    dict: A JSON response indicating the service's status.
    """
    return {"message": "Welcome to RAG Service"}