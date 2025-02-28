import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TEMPERATURE = 0.7
DEFAULT_CALLS = 1

def get_api_key():
    return os.getenv("AZURE_OPENAI_API_KEY")

def get_api_endpoint():
    return os.getenv("AZURE_OPENAI_ENDPOINT")