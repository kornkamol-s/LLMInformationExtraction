from dotenv import load_dotenv
import os, json

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
VECTOR_STORE_DIR = 'log/vector-store'

with open('config/question_mapping.json', 'r') as f:
    QUESTION_MAPPING = json.load(f)

with open('config/heading_mapping.json', 'r') as f:
    HEADING_MAPPING = json.load(f)
