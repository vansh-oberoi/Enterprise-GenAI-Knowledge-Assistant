import os
from dotenv import load_dotenv

load_dotenv()
CHROMA_DIR = os.getenv('CHROMA_DIR', './chroma_db')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '900'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '150'))
TOP_K = int(os.getenv('TOP_K', '5'))

# Ollama runs locally and exposes an OpenAI-compatible API. Non-empty .env
# values override these defaults; blank values fall back to local Ollama.
LLM_BASE_URL = os.getenv('LLM_BASE_URL', '').strip() or 'http://localhost:11434/v1'
LLM_API_KEY = os.getenv('LLM_API_KEY', '').strip() or 'ollama'
LLM_MODEL = os.getenv('LLM_MODEL', '').strip() or 'qwen2.5:3b'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
