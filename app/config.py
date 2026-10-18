import os
from dotenv import load_dotenv

load_dotenv()
CHROMA_DIR = os.getenv('CHROMA_DIR', './chroma_db')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '900'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '150'))
TOP_K = int(os.getenv('TOP_K', '5'))
LLM_BASE_URL = os.getenv('LLM_BASE_URL', '').strip()
LLM_API_KEY = os.getenv('LLM_API_KEY', '').strip()
LLM_MODEL = os.getenv('LLM_MODEL', '').strip()
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
