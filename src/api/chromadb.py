import os
import dotenv
from posthog import api_key

import chromadb

dotenv.load_dotenv('../.env.local')

client = chromadb.Client(
    api_key = os.getenv('CHROMA_API_KEY'),
    tenant = os.getenv('TENANT'),
    database = os.getenv('DATABASE')
)


