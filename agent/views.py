from django.http import StreamingHttpResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
from agent.llm.rag_chain import create_rag_chain
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

rag_chain = create_rag_chain() # Initialize the RAG chain

def chat_ui(request):
    return render(request, "chat_ui.html")

@csrf_exempt # Disable CSRF protection for simplicity (not recommended for production)
def chat_api(request):
    if request.method == "POST":
        # Check Content-Type header first
        if request.content_type != 'application/json':
            logging.error(f"Invalid Content-Type: Expected 'application/json', got '{request.content_type}'")
            return HttpResponseBadRequest(f"Invalid Content-Type: Expected 'application/json', got '{request.content_type}'")

        try:
            raw_body = request.body.decode('utf-8') # Decode first
            logging.info(f"Received raw body: {raw_body}") # Log the raw body
            data = json.loads(raw_body) # Then parse

            # Use .get() with a check for None to distinguish missing key from empty string
            user_prompt = data.get("prompt")
            if user_prompt is None: # Check if the key exists, even if value is ""
                logging.error("Missing 'prompt' in request body")
                return HttpResponseBadRequest("Missing 'prompt' in request body")

            # Get chat history, default to empty list if not provided or not a list
            chat_history = data.get("history", [])
            if not isinstance(chat_history, list):
                logging.warning(f"Received invalid 'history' format (expected list, got {type(chat_history)}). Using empty history.")
                chat_history = []

            logging.info(f"Received prompt: {user_prompt}")
            logging.info(f"Received history: {chat_history}")

            def stream():
                try:
                    # Pass both question and history to the RAG chain
                    input_data = {"question": user_prompt, "chat_history": chat_history}
                    logging.info(f"Streaming data to RAG chain: {input_data}")
                    for chunk in rag_chain.stream(input_data):
                        # Assuming the chunk structure remains the same or is handled by the chain
                        # If the chain output structure changes with history, adjust here.
                        logging.info(f"Yielding chunk: {chunk}")
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n" # SSE format
                except Exception as e:
                    logging.error(f"Error during RAG chain execution: {e}", exc_info=True)
                    yield f"data: {json.dumps({'error': 'Internal server error'})}\n\n"
                finally:
                    yield "data: [DONE]\n\n"

            return StreamingHttpResponse(stream(), content_type='text/event-stream')
        except json.JSONDecodeError as e:
            # Log the error and the raw body content for debugging
            logging.error(f"JSONDecodeError: {e} - Raw body was: {request.body.decode('utf-8', errors='ignore')}")
            return HttpResponseBadRequest("Invalid JSON in request body")
        except Exception as e: # Catch other potential errors
            logging.error(f"Unexpected error in chat_api: {e}", exc_info=True)
            return HttpResponseBadRequest("An unexpected error occurred")
    else:
        return HttpResponseBadRequest("Only POST requests are allowed")
