import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
DATA_PATH = "agent/data"

# Determine Qdrant URL based on environment
RUNNING_IN_DOCKER = os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true"
QDRANT_URL = "http://qdrant:6333" if RUNNING_IN_DOCKER else "http://localhost:6333"
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # Uncomment if you set an API key for Qdrant
QDRANT_COLLECTION_NAME = "flixbot_rag_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def load_documents(data_path):
    """Loads all PDF documents from the specified directory."""
    pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
    if not pdf_files:
        logging.warning(f"No PDF files found in {data_path}")
        return []
    
    docs = []
    logging.info(f"Found {len(pdf_files)} PDF files to load.")
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())
            logging.info(f"Successfully loaded {pdf_path}")
        except Exception as e:
            logging.error(f"Error loading {pdf_path}: {e}")
    return docs

def split_documents(docs):
    """Splits documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(docs)
    logging.info(f"Split {len(docs)} documents into {len(chunks)} chunks.")
    return chunks

def get_embeddings_model():
    """Initializes and returns the embeddings model."""
    logging.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    # You might need to configure device='cuda' if GPU is available and configured
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return embeddings

def ingest_data():
    """Main function to run the data ingestion pipeline."""
    logging.info("Starting data ingestion process...")

    # 1. Load documents
    documents = load_documents(DATA_PATH)
    if not documents:
        logging.warning("No documents loaded. Exiting ingestion process.")
        return

    # 2. Split documents
    chunks = split_documents(documents)
    if not chunks:
        logging.warning("No chunks created. Exiting ingestion process.")
        return

    # 3. Initialize embeddings model
    embeddings = get_embeddings_model()

    # 4. Initialize Qdrant client and ingest data
    logging.info(f"Initializing Qdrant client for collection: {QDRANT_COLLECTION_NAME} at {QDRANT_URL}")
    try:
        # First verify Qdrant connection
        qdrant_client = QdrantClient(QDRANT_URL)
        qdrant_client.get_collections()
        logging.info("Successfully connected to Qdrant server")
        
        # Then ingest documents
        qdrant = Qdrant.from_documents(
            chunks,
            embeddings,
            url=QDRANT_URL,
            prefer_grpc=False,
            collection_name=QDRANT_COLLECTION_NAME,
            force_recreate=True,
        )
        logging.info(f"Successfully ingested {len(chunks)} chunks into Qdrant collection '{QDRANT_COLLECTION_NAME}'.")
    except Exception as e:
        logging.error(f"Failed to connect/ingest data into Qdrant: {e}")
        logging.error("Please ensure:")
        logging.error(f"1. Qdrant is running at {QDRANT_URL}")
        logging.error("2. The container network is properly configured if running in Docker")
        logging.error("3. The ports are correctly exposed in docker-compose.yml")

if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        logging.info(f"Created data directory: {DATA_PATH}")
        logging.warning("Data directory was empty. Please add PDF files to agent/data/ before running ingestion.")
    
    ingest_data()
