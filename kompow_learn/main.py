import os
import sys
from dotenv import load_dotenv
import json
from datetime import datetime, timezone

# Add project root to sys.path for local imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.knowledge_base import get_user_knowledge_base, add_flashcard_set_to_kb, query_knowledge_base
from agno_agents.profile_agent import LearningProfileAgent
from agno_agents.research_agent import ResearchAgent
from agno_agents.flashcard_agent import FlashcardGenerationAgent

# --- Environment Variable Loading and Crucial Checks ---
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
    print("CRITICAL ERROR: OPENAI_API_KEY is not set or is a placeholder in your .env file.")
    print("The KompowLearn main processing pipeline cannot function without a valid OpenAI API key.")
    print("Please set it in the .env file at the root of the project and restart.")
    sys.exit(1) # Exit if API key is not configured

# --- Orchestration Function ---
def process_user_content_and_generate_flashcards(user_id: str):
    """
    Orchestrates the process of:
    1. Analyzing a user's existing knowledge base content to create a learning profile.
    2. Researching topics identified in the profile.
    3. Generating flashcards from the researched content.
    4. Storing these new flashcards back into the user's knowledge base.
    """
    print(f"\n=== Starting Content Processing Pipeline for User: {user_id} ===")

    # --- Step 1: Initialize Agents ---
    print("\n--- Step 1: Initializing AI Agents ---")
    try:
        profile_agent = LearningProfileAgent(user_id=user_id)
        research_agent = ResearchAgent() # Does not require user_id at init
        flashcard_agent = FlashcardGenerationAgent() # Does not require user_id at init
        print("AI Agents initialized successfully.")
    except ValueError as ve:
        print(f"ERROR: Failed to initialize AI agents: {ve}")
        print("Please ensure your OPENAI_API_KEY is correctly configured in .env.")
        return
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during agent initialization: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Step 2: Analyze Learning Profile from User's Existing KB Content ---
    print(f"\n--- Step 2: Analyzing Learning Profile for User '{user_id}' ---")
    try:
        profile_analysis_text = profile_agent.analyze_user_profile(max_docs=50, query_str="")

        if not profile_analysis_text or "Cannot analyze profile" in profile_analysis_text or "No documents found" in profile_analysis_text:
            print(f"INFO: Learning profile analysis for user '{user_id}' did not yield significant topics or content.")
            print(f"Raw analysis output snippet: {profile_analysis_text[:500] if profile_analysis_text else 'N/A'}")
            print("Skipping further processing for this user as no clear topics were derived from profile analysis.")
            return
        print(f"Learning Profile Analysis (first 300 chars): \n{profile_analysis_text[:300]}...")
    except Exception as e:
        print(f"ERROR: Failed during learning profile analysis for user '{user_id}': {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Step 3: Research Identified Topics ---
    print("\n--- Step 3: Researching Identified Topics from Profile Analysis ---")
    try:
        researched_text = research_agent.research_topics(profile_analysis_text) # Pass the string from profile analysis

        if not researched_text or "Failed to conduct research" in researched_text or len(researched_text) < 100:
            print(f"INFO: Research based on profile analysis for user '{user_id}' did not yield sufficient content.")
            print(f"Raw research output snippet: {researched_text[:500] if researched_text else 'N/A'}")
            print("Skipping flashcard generation as research content is inadequate.")
            return
        print(f"Researched Text (first 300 chars): \n{researched_text[:300]}...")
    except Exception as e:
        print(f"ERROR: Failed during research phase for user '{user_id}': {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Step 4: Generate Flashcards from Researched Text ---
    print("\n--- Step 4: Generating Flashcards from Researched Text ---")
    try:
        generated_flashcards_output = flashcard_agent.generate_flashcards_from_text(researched_text)

        if not isinstance(generated_flashcards_output, list) or not generated_flashcards_output:
            print(f"INFO: Flashcard generation for user '{user_id}' did not produce a valid list of flashcards.")
            print(f"Raw flashcard agent output: {generated_flashcards_output}")
            print("Skipping storage of flashcards.")
            return
        print(f"Successfully generated {len(generated_flashcards_output)} flashcards.")
    except Exception as e:
        print(f"ERROR: Failed during flashcard generation for user '{user_id}': {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Step 5: Store Generated Flashcards in User's Knowledge Base ---
    print("\n--- Step 5: Storing Generated Flashcards ---")
    kb_user = get_user_knowledge_base(user_id)
    if not kb_user:
        # This might happen if get_user_knowledge_base had an unrecoverable error specific to this user
        # not caught by the general API key check (e.g., filesystem permission for LanceDB URI).
        print(f"ERROR: Could not initialize or retrieve Knowledge Base for user '{user_id}'. Cannot store flashcards.")
        return

    try:
        flashcards_json_str = json.dumps(generated_flashcards_output)
        topic_for_flashcards = f"Automated Flashcards from Profile Update - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"

        store_success = add_flashcard_set_to_kb(
            kb=kb_user, user_id=user_id, topic=topic_for_flashcards,
            flashcards_json_string=flashcards_json_str, source="automated_pipeline_from_profile"
        )

        if store_success:
            print(f"Flashcard set successfully stored for user '{user_id}' with topic '{topic_for_flashcards}'.")
        else:
            print(f"Failed to store flashcard set for user '{user_id}'. Check logs from knowledge_base.py. This might be due to an issue with adding to the KB even if the API key was present for agents (e.g. disk space, permissions, or schema issues if adding non-embedded data to a vector-expecting table without a valid embedding).")

    except json.JSONDecodeError as jde:
        print(f"ERROR: Internal error - generated flashcards could not be serialized to JSON: {jde}")
    except Exception as e:
        print(f"ERROR: Failed during storage of flashcards for user '{user_id}': {e}")
        import traceback
        traceback.print_exc()

    print(f"\n=== Completed Content Processing Pipeline for User: {user_id} ===")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("KompowLearn Main Orchestration Script - Manual Trigger Mode")
    print("-----------------------------------------------------------")

    test_user_id_main = os.getenv("EMAIL_USER") # e.g., your email if you've parsed its content into KB
    if not test_user_id_main:
        print("ERROR: EMAIL_USER not set in .env file. Please set it to a valid user_id.")
        print("This user_id should correspond to a knowledge base that has some initial content for the ProfileAgent to analyze.")
        sys.exit(1)

    print(f"Target User ID for processing: {test_user_id_main}")
    print("This script will attempt to run the full agent pipeline: Profile -> Research -> Flashcards -> Storage.")
    print("Prerequisites:")
    print(f"  1. Valid OPENAI_API_KEY in .env (already checked).")
    print(f"  2. User '{test_user_id_main}' should have existing content in their Knowledge Base ")
    print(f"     (e.g., from running `python utils/email_parser.py` with this user as EMAIL_USER).")
    print(f"     The directory 'tmp/lancedb_store/user_{sanitize_table_name(test_user_id_main)}' should ideally exist and contain data.")

    kb_check = get_user_knowledge_base(test_user_id_main) # from utils.knowledge_base import sanitize_table_name if using it here
                                                       # sanitize_table_name is not directly used in this print, but good to remember for path construction.
    if kb_check:
        if kb_check.vector_db.embedder: # Check if embedder is available (API key was valid for KB init)
            print(f"Checking if user '{test_user_id_main}' KB has any existing content (requires valid API key for this check)...")
            try:
                # A simple query to check for any content.
                # This is just an indicative check. ProfileAgent does its own fetching.
                existing_docs_sample = query_knowledge_base(kb_check, query_text="*", limit=1) # Using "*" as a very broad query
                if existing_docs_sample:
                    print(f"User '{test_user_id_main}' KB appears to have content. Proceeding with pipeline.")
                else:
                    print(f"Warning: User '{test_user_id_main}' KB might be empty or content not found with generic query '*'. Profile analysis may yield no topics.")
            except Exception as e_kb_check:
                 print(f"Notice: Could not perform a full pre-check of user KB content due to: {e_kb_check}. The Profile Agent will still attempt to load documents.")
        else:
            print("Notice: OPENAI_API_KEY is missing or invalid for Knowledge Base embedder initialization. ")
            print("The ProfileAgent's ability to analyze existing content will be severely limited or fail.")
            print("The pipeline will proceed, but expect issues in the Profile Analysis step if it relies on semantic search of existing KB content.")
    else:
        print(f"Warning: Could not initialize Knowledge Base for user '{test_user_id_main}'. The pipeline will likely fail early.")

    process_user_content_and_generate_flashcards(test_user_id_main)

    print("\n-----------------------------------------------------------")
    print("KompowLearn Main Orchestration Script - Execution Finished.")
