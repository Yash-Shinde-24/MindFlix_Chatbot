import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from qdrant_client import QdrantClient, models

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# --- Constants ---
# Determine Qdrant URL based on environment
IS_DOCKER = os.getenv("RUNNING_IN_DOCKER", "false").lower() == "true"
QDRANT_HOST = "qdrant" if IS_DOCKER else "localhost"
QDRANT_PORT = 6333
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # Uncomment if needed
QDRANT_COLLECTION_NAME = "flixbot_rag_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-specdec") # Default model
SYSTEM_PROMPT_TEMPLATE = """You are a smart travel assistant working for a travel company called FlixBot Travels.
Use the following pieces of retrieved context to answer the user's question.
If you don't know the answer, try to provide a helpful response based on the available information.
Keep the answer concise and directly related to the context provided.
Always provide accurate information based *only* on the context. If the context doesn't contain the answer, say so.
Do not mention that you are referencing context in your response.

If the question is about booking policies, please summarize the policies based on the context.

Context:
{context}"""

# --- Initialization ---

def get_vector_store():
    """Initializes and returns the Qdrant vector store client."""
    if not GROQ_API_KEY:
        logging.error("GROQ_API_KEY not found in environment variables.")
        raise ValueError("GROQ_API_KEY must be set.")

    logging.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    logging.info(f"Connecting to Qdrant vector store: {QDRANT_COLLECTION_NAME} at {QDRANT_URL}")
    try:
        client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
        logging.info("Successfully connected to Qdrant.")

        # Initialize vectorstore
        vector_store = Qdrant(
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            embeddings=embeddings,  # Pass embeddings object directly
        )
        return vector_store
    except Exception as e:
        logging.error(f"Failed to connect to Qdrant collection '{QDRANT_COLLECTION_NAME}': {e}")
        logging.error("Ensure the Qdrant service is running and the collection exists (run ingest_data.py first).")
        raise e

def get_llm():
    """Initializes and returns the Groq LLM."""
    logging.info(f"Initializing Groq LLM with model: {GROQ_MODEL_NAME}")
    llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL_NAME)
    return llm

def format_docs(docs):
    """Formats retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- RAG Chain Definition ---

def create_rag_chain():
    """Creates and returns the full RAG chain."""
    vector_store = get_vector_store()
    llm = get_llm()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Define the prompt structure, adding a placeholder for chat history
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"), # Add placeholder for history
            ("human", "{question}"),
        ]
    )

    # Define the RAG chain using LCEL
    # The input to the chain is expected to be a dictionary with "question" and "chat_history"
    def retrieve_context(input_dict):
        """Retrieves documents based on the question."""
        question = input_dict["question"]
        retrieved_docs = retriever.get_relevant_documents(question)
        formatted_docs = format_docs(retrieved_docs)
        logging.info(f"Retrieved context for question '{question}':\n{formatted_docs[:500]}...") # Log snippet
        return formatted_docs

    # Chain takes the input dict, retrieves context based on 'question',
    # and passes 'context', 'question', and 'chat_history' to the prompt.
    rag_chain = (
        RunnablePassthrough.assign(context=retrieve_context) # Assign context based on question
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("RAG chain created successfully.")
    return rag_chain

# --- Main execution (for testing) ---
if __name__ == "__main__":
    # This part is for basic testing of the chain setup
    # It requires Qdrant to be running and the collection to be populated
    logging.info("Testing RAG chain setup...")
    try:
        rag_chain = create_rag_chain()
        logging.info("RAG Chain instance created.")
        
    except Exception as e:
        logging.error(f"Error during RAG chain test setup: {e}")
