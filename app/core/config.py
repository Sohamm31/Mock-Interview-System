# app/core/config.py
import os
import logging
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# --- Database Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set!")
    raise ValueError("DATABASE_URL environment variable not set. Please configure it in your .env file or environment.")
logger.info(f"Config: DATABASE_URL loaded.")

# --- JWT Configuration ---
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    logger.warning("SECRET_KEY environment variable not set! Using a default, which is INSECURE for production.")
    SECRET_KEY = "insecure-default-secret-key-please-change-this" # Fallback for development
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300 # 5 hours

# --- LLM and Embeddings Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "") # Default to empty if not set

# Initialize LLM (Using ChatOpenAI for OpenRouter)
llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1-0528-qwen3-8b:free", # Specific model as requested
    openai_api_key=OPENROUTER_API_KEY, # Use the API key from .env
    openai_api_base="https://openrouter.ai/api/v1", # OpenRouter API base URL
    temperature=0.7 # Keep temperature as before
)
logger.info("Config: LangChain LLM (ChatOpenAI with OpenRouter/DeepSeek) initialized.")

# Initialize Embeddings (HuggingFace sentence-transformers/all-MiniLM-L6-v2)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
logger.info("Config: HuggingFace Embeddings model loaded.")
