import os
import sys
from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn
import json
import traceback
from typing import List, Optional # For type hinting

# Add project root to sys.path for local imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agno_agents.research_agent import ResearchAgent
from agno_agents.flashcard_agent import FlashcardGenerationAgent
from utils.knowledge_base import (
    get_user_knowledge_base,
    add_flashcard_set_to_kb,
    get_flashcard_sets_for_user, # Added
    get_available_flashcard_topics # Added
)

# --- Load Environment Variables ---
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Global Variables & Agent Initialization ---
OPENAI_API_KEY_GLOBAL = os.getenv("OPENAI_API_KEY")
research_agent_api: ResearchAgent | None = None
flashcard_agent_api: FlashcardGenerationAgent | None = None
# KB for API user is initialized once, assuming one main KB for API generated content or use user_id from request.
# For retrieving, we'll get KB based on user_id in request.
ON_DEMAND_API_USER_ID = "api_on_demand_user" # Default user for storing API-generated cards

initialization_fatal_error = False
critical_error_messages = []

if not OPENAI_API_KEY_GLOBAL or OPENAI_API_KEY_GLOBAL == "your_openai_api_key_here" or len(OPENAI_API_KEY_GLOBAL) < 10:
    msg = "CRITICAL API SERVER ERROR: OPENAI_API_KEY is not set or is invalid in .env file. AI Agents and KB operations requiring embeddings will fail."
    print(msg)
    critical_error_messages.append(msg)
    initialization_fatal_error = True # If API key is bad, most functionalities are impacted.
else:
    try:
        print("Initializing AI Agents for API Server...")
        research_agent_api = ResearchAgent(model_id="gpt-3.5-turbo")
        flashcard_agent_api = FlashcardGenerationAgent(model_id="gpt-3.5-turbo-1106")
        print("AI Agents initialized successfully.")
    except ValueError as ve:
        msg = f"CRITICAL API SERVER ERROR during agent initialization: {ve}"
        print(msg)
        critical_error_messages.append(msg)
        initialization_fatal_error = True
    except Exception as e:
        msg = f"CRITICAL API SERVER ERROR: An unexpected error during agent/KB initialization: {e}"
        print(msg)
        traceback.print_exc()
        critical_error_messages.append(msg)
        initialization_fatal_error = True

# Initialize a default KB for API-generated content storage if needed, but retrieval will use requested user_id
# This kb_for_api_user is for the POST endpoint if it stores to a generic API user.
kb_for_api_user = get_user_knowledge_base(ON_DEMAND_API_USER_ID)
if not kb_for_api_user:
    msg = f"WARNING: Default KB for API user '{ON_DEMAND_API_USER_ID}' could NOT be initialized. Storing API-generated flashcards might fail if this default KB is used."
    print(msg)
    critical_error_messages.append(msg) # Add to messages, but might not be fatal for all endpoints

app = FastAPI(
    title="KompowLearn API",
    description="API for generating and retrieving educational content like flashcards using AI agents.",
    version="0.2.0" # Incremented version
)

# --- Pydantic Models ---
class TopicRequest(BaseModel):
    topic: str = Field(..., min_length=3, example="Basics of Quantum Computing")

class Flashcard(BaseModel):
    question: str
    answer: str

class FlashcardResponse(BaseModel):
    topic: str
    flashcards: List[Flashcard] | None = Field(None, description="List of generated flashcards.")
    message: str = Field(description="A message indicating the outcome.")
    researched_text_snippet: str | None = Field(None, description="Optional snippet of researched text.")
    storage_status: str | None = Field(None, description="Status of storing flashcards in KB.")

class StoredFlashcardSet(BaseModel):
    document_id: str
    topic: str
    creation_date: str
    source: str
    flashcards: List[Flashcard]

class StoredFlashcardsResponse(BaseModel):
    user_id: str
    topic_filter: Optional[str] = None
    retrieved_flashcard_sets: List[StoredFlashcardSet]
    message: str

class TopicListResponse(BaseModel):
    user_id: str
    topics: List[str]
    message: str

class ErrorDetail(BaseModel):
    detail: str

# --- API Endpoints ---

@app.get("/", summary="Root", tags=["General"])
async def root():
    health_status = "degraded_due_to_init_error: " + " | ".join(critical_error_messages) if critical_error_messages else "running"
    return {
        "message": "Welcome to the KompowLearn API!",
        "documentation": "/docs",
        "health": health_status
    }

@app.post(
    "/generate-flashcards/", response_model=FlashcardResponse,
    responses={ 503: {"model": ErrorDetail}, 500: {"model": ErrorDetail}, 400: {"model": ErrorDetail}},
    summary="Generate Flashcards for a Topic", tags=["Flashcards"]
)
async def api_generate_flashcards(request: TopicRequest):
    if initialization_fatal_error or not research_agent_api or not flashcard_agent_api:
        raise HTTPException(status_code=503, detail=f"Service Unavailable: AI Agents or critical components not initialized. Errors: {' | '.join(critical_error_messages)}")
    topic = request.topic
    print(f"API: Request to generate flashcards for topic: '{topic}'")
    try:
        researched_text = research_agent_api.research_topics(topic)
        if not researched_text or "Failed to conduct research" in researched_text or len(researched_text) < 50:
            raise HTTPException(status_code=500, detail=f"Research failed or insufficient content for topic: {topic}. Output: {researched_text}")
        snippet = researched_text[:200] + "..."
        generated_output = flashcard_agent_api.generate_flashcards_from_text(researched_text)
        if not isinstance(generated_output, list) or not generated_output:
            raise HTTPException(status_code=500, detail=f"Flashcard generation failed for topic '{topic}'. Agent output: {generated_output}")

        storage_msg = "Storage skipped: Default KB for API user not available."
        if kb_for_api_user: # Using the globally initialized KB for ON_DEMAND_API_USER_ID
            try:
                flashcards_json_str = json.dumps(generated_output)
                store_success = add_flashcard_set_to_kb(kb_for_api_user, ON_DEMAND_API_USER_ID, topic, flashcards_json_str, source="api_on_demand_generation")
                storage_msg = "Flashcards stored successfully." if store_success else "Failed to store flashcards in KB."
            except Exception as e_store: storage_msg = f"Error during flashcard storage: {e_store}"

        return FlashcardResponse(topic=topic, flashcards=generated_output, message="Flashcards generated.", researched_text_snippet=snippet, storage_status=storage_msg)
    except HTTPException: raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected server error for topic '{topic}': {e}")


@app.get(
    "/retrieve-flashcards/", response_model=StoredFlashcardsResponse,
    responses={503: {"model": ErrorDetail}, 404: {"model": ErrorDetail}, 500: {"model": ErrorDetail}},
    summary="Retrieve Stored Flashcard Sets for a User", tags=["Flashcards"]
)
async def api_retrieve_flashcards(user_id: str = Query(..., description="User ID for whom to retrieve flashcards."),
                                  topic: Optional[str] = Query(None, description="Optional topic to filter flashcards.")):
    print(f"API: Request to retrieve flashcards for user_id: '{user_id}', topic: '{topic if topic else 'any'}'.")
    # API key check is for embedder, which get_flashcard_sets_for_user needs for semantic search.
    if initialization_fatal_error and "OPENAI_API_KEY" in " ".join(critical_error_messages) : # Check if the specific error is API key related
         raise HTTPException(status_code=503, detail=f"Service Unavailable: OpenAI API Key issue prevents KB search. Errors: {' | '.join(critical_error_messages)}")

    kb_user_retrieve = get_user_knowledge_base(user_id)
    if not kb_user_retrieve:
        raise HTTPException(status_code=503, detail=f"Knowledge Base for user '{user_id}' could not be initialized.")
    if not kb_user_retrieve.vector_db.embedder: # Crucial check for current implementation of get_flashcard_sets
        raise HTTPException(status_code=503, detail=f"Knowledge Base for user '{user_id}' lacks a functional embedder (likely API key issue). Cannot retrieve flashcard sets with current method.")

    try:
        flashcard_documents = get_flashcard_sets_for_user(kb_user_retrieve, user_id=user_id, topic=topic, limit=20)
        processed_sets: List[StoredFlashcardSet] = []
        if not flashcard_documents:
            return StoredFlashcardsResponse(user_id=user_id, topic_filter=topic, retrieved_flashcard_sets=[], message="No flashcard sets found for the given criteria.")

        for doc in flashcard_documents:
            try:
                parsed_flashcards_list = json.loads(doc.content)
                # Ensure each item in parsed_flashcards_list is a dict with 'question' and 'answer'
                valid_flashcards_for_set = [Flashcard(**fc) for fc in parsed_flashcards_list if isinstance(fc, dict) and 'question' in fc and 'answer' in fc]

                if valid_flashcards_for_set: # Only include if there are valid flashcards after parsing
                    processed_sets.append(StoredFlashcardSet(
                        document_id=doc.id,
                        topic=doc.metadata.get("topic", "Unknown Topic"),
                        creation_date=doc.metadata.get("creation_date", "N/A"),
                        source=doc.metadata.get("source", "N/A"),
                        flashcards=valid_flashcards_for_set
                    ))
            except json.JSONDecodeError:
                print(f"API WARNING: Failed to parse JSON content for document ID {doc.id} for user {user_id}.")
                # Optionally include this info in the response message or skip the corrupted doc

        return StoredFlashcardsResponse(user_id=user_id, topic_filter=topic, retrieved_flashcard_sets=processed_sets, message=f"Retrieved {len(processed_sets)} flashcard set(s).")
    except HTTPException: raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected server error retrieving flashcards for user '{user_id}': {e}")


@app.get(
    "/list-flashcard-topics/", response_model=TopicListResponse,
    responses={503: {"model": ErrorDetail}, 500: {"model": ErrorDetail}},
    summary="List Available Flashcard Topics for a User", tags=["Flashcards"]
)
async def api_list_flashcard_topics(user_id: str = Query(..., description="User ID for whom to list topics.")):
    print(f"API: Request to list flashcard topics for user_id: '{user_id}'.")
    if initialization_fatal_error and "OPENAI_API_KEY" in " ".join(critical_error_messages):
         raise HTTPException(status_code=503, detail=f"Service Unavailable: OpenAI API Key issue prevents KB search. Errors: {' | '.join(critical_error_messages)}")

    kb_user_topics = get_user_knowledge_base(user_id)
    if not kb_user_topics:
        raise HTTPException(status_code=503, detail=f"Knowledge Base for user '{user_id}' could not be initialized.")
    if not kb_user_topics.vector_db.embedder:
        raise HTTPException(status_code=503, detail=f"Knowledge Base for user '{user_id}' lacks a functional embedder (API key issue). Cannot list topics with current method.")

    try:
        topics = get_available_flashcard_topics(kb_user_topics, user_id)
        return TopicListResponse(user_id=user_id, topics=topics, message=f"Found {len(topics)} topic(s).")
    except HTTPException: raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected server error listing topics for user '{user_id}': {e}")


# --- Uvicorn Runner ---
if __name__ == "__main__":
    if initialization_fatal_error: # This flag is set if API key is bad or agents failed init
        print("*********************************************************************************")
        print(" FastAPI server WILL NOT START due to critical initialization errors.")
        print(" Errors encountered: ")
        for err_msg in critical_error_messages:
            print(f"  - {err_msg}")
        print(" Please check your .env configuration (especially OPENAI_API_KEY) and try again.")
        print("*********************************************************************************")
        sys.exit(1)

    print("Starting Uvicorn server for KompowLearn API at http://0.0.0.0:8000")
    print("Access API documentation at http://0.0.0.0:8000/docs")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
