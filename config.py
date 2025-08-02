import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FINE_TUNED_MODEL = "ft:gpt-3.5-turbo-1106:personal::Byb7huHp"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "models/embedding-001"
EMBEDDING_DIMENSION = 768  # Gemini embedding dimension


CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "insurance-indexs")
