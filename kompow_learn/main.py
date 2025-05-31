import os
import sys
from dotenv import load_dotenv
import json
from datetime import datetime, timezone
import time # For polling loop

# Add project root to sys.path for local imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.knowledge_base import get_user_knowledge_base, add_flashcard_set_to_kb, query_knowledge_base
from utils.email_parser import process_and_store_emails, connect_to_mailbox # Added email_parser imports
from agno_agents.profile_agent import LearningProfileAgent
from agno_agents.research_agent import ResearchAgent
from agno_agents.flashcard_agent import FlashcardGenerationAgent

# --- Environment Variable Loading and Crucial Checks ---
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
POLLING_INTERVAL_SECONDS = int(os.getenv("POLLING_INTERVAL_SECONDS", "300")) # Default to 5 mins

abort_execution = False
if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
    print("CRITICAL ERROR: OPENAI_API_KEY is not set or is a placeholder in your .env file.")
    abort_execution = True
if not EMAIL_HOST or EMAIL_HOST == "your_imap_server.com":
    print("CRITICAL ERROR: EMAIL_HOST is not set or is a placeholder in your .env file.")
    abort_execution = True
if not EMAIL_USER or EMAIL_USER == "your_email@example.com":
    print("CRITICAL ERROR: EMAIL_USER is not set or is a placeholder in your .env file.")
    abort_execution = True
if not EMAIL_PASS or EMAIL_PASS == "your_password":
    print("CRITICAL ERROR: EMAIL_PASS is not set or is a placeholder in your .env file.")
    abort_execution = True

if abort_execution:
    print("One or more critical environment variables are missing or placeholders.")
    print("The KompowLearn application cannot function without these.")
    print("Please set them in the .env file at the root of the project and restart.")
    sys.exit(1)

# --- Orchestration Function ---
def process_user_content_and_generate_flashcards(user_id: str):
    print(f"\n=== Starting Content Processing Pipeline for User: {user_id} ===")
    try:
        print("\n--- Step 1: Initializing AI Agents ---")
        profile_agent = LearningProfileAgent(user_id=user_id)
        research_agent = ResearchAgent()
        flashcard_agent = FlashcardGenerationAgent()
        print("AI Agents initialized successfully.")
    except ValueError as ve:
        print(f"ERROR: Failed to initialize AI agents for {user_id}: {ve}")
        return
    except Exception as e:
        print(f"ERROR: An unexpected error during agent initialization for {user_id}: {e}")
        import traceback; traceback.print_exc()
        return

    print(f"\n--- Step 2: Analyzing Learning Profile for User '{user_id}' ---")
    try:
        profile_analysis_text = profile_agent.analyze_user_profile(max_docs=50, query_str="")
        if not profile_analysis_text or "Cannot analyze profile" in profile_analysis_text or "No documents found" in profile_analysis_text:
            print(f"INFO: Learning profile analysis for '{user_id}' did not yield topics. Output: {profile_analysis_text if profile_analysis_text else 'N/A'}")
            return
        print(f"Learning Profile Analysis for '{user_id}' (first 300 chars): \n{profile_analysis_text[:300]}...")
    except Exception as e:
        print(f"ERROR: Failed profile analysis for '{user_id}': {e}")
        import traceback; traceback.print_exc()
        return

    print("\n--- Step 3: Researching Identified Topics from Profile Analysis ---")
    try:
        researched_text = research_agent.research_topics(profile_analysis_text)
        if not researched_text or "Failed to conduct research" in researched_text or len(researched_text) < 100:
            print(f"INFO: Research for '{user_id}' did not yield sufficient content. Output: {researched_text if researched_text else 'N/A'}")
            return
        print(f"Researched Text for '{user_id}' (first 300 chars): \n{researched_text[:300]}...")
    except Exception as e:
        print(f"ERROR: Failed research phase for '{user_id}': {e}")
        import traceback; traceback.print_exc()
        return

    print("\n--- Step 4: Generating Flashcards from Researched Text ---")
    try:
        generated_flashcards_output = flashcard_agent.generate_flashcards_from_text(researched_text)
        if not isinstance(generated_flashcards_output, list) or not generated_flashcards_output:
            print(f"INFO: Flashcard generation for '{user_id}' failed. Output: {generated_flashcards_output}")
            return
        print(f"Successfully generated {len(generated_flashcards_output)} flashcards for '{user_id}'.")
    except Exception as e:
        print(f"ERROR: Failed flashcard generation for '{user_id}': {e}")
        import traceback; traceback.print_exc()
        return

    print("\n--- Step 5: Storing Generated Flashcards ---")
    kb_user = get_user_knowledge_base(user_id)
    if not kb_user:
        print(f"ERROR: Could not initialize KB for '{user_id}'. Cannot store flashcards.")
        return
    try:
        flashcards_json_str = json.dumps(generated_flashcards_output)
        topic_for_flashcards = f"Automated Flashcards from Profile Update - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
        store_success = add_flashcard_set_to_kb(
            kb=kb_user, user_id=user_id, topic=topic_for_flashcards,
            flashcards_json_string=flashcards_json_str, source="automated_pipeline_from_profile_update"
        )
        if store_success: print(f"Flashcard set stored for '{user_id}', topic '{topic_for_flashcards}'.")
        else: print(f"Failed to store flashcard set for '{user_id}'. See KB logs.")
    except Exception as e:
        print(f"ERROR: Failed storing flashcards for '{user_id}': {e}")
        import traceback; traceback.print_exc()
    print(f"\n=== Completed Content Processing Pipeline for User: {user_id} ===")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("===========================================================")
    print("    KompowLearn - Continuous Email Polling & Processing    ")
    print("===========================================================")
    print(f"Starting up at {datetime.now(timezone.utc).isoformat()}...")
    print(f"Checking for new emails every {POLLING_INTERVAL_SECONDS} seconds.")
    print("Press Ctrl+C to exit gracefully.")

    run_count = 0
    try:
        while True:
            run_count += 1
            print(f"\n--- Polling Cycle #{run_count} [{datetime.now(timezone.utc).isoformat()}] ---")
            print("Attempting to connect to mailbox...")
            mail_server = connect_to_mailbox(EMAIL_HOST, EMAIL_USER, EMAIL_PASS)

            if mail_server:
                print("Successfully connected to mailbox. Processing emails...")
                try:
                    # Process emails and get user_ids whose KBs were updated
                    # default_user_id_if_no_sender is used if email 'From' field is weird/missing.
                    # For emails *to* EMAIL_USER, the sender is the one whose KB is updated.
                    updated_user_ids = process_and_store_emails(
                        mail_server,
                        default_user_id_if_no_sender="unknown_sender_user",
                        max_emails_to_process_cycle=10 # Limit emails per poll
                    )
                except Exception as e_process:
                    print(f"ERROR: An error occurred during email processing: {e_process}")
                    import traceback; traceback.print_exc()
                    updated_user_ids = [] # Ensure it's an empty list on error
                finally:
                    try:
                        mail_server.logout()
                        print("Mailbox connection closed.")
                    except Exception as e_logout:
                        print(f"Warning: Error during mailbox logout: {e_logout}")

                if updated_user_ids:
                    print(f"Knowledge bases updated for users: {updated_user_ids}. Triggering agent processing for each.")
                    for user_id_to_process in updated_user_ids:
                        # It's important that process_user_content_and_generate_flashcards
                        # is robust and handles its own errors for each user,
                        # so one user failing doesn't stop processing for others.
                        try:
                            process_user_content_and_generate_flashcards(user_id_to_process)
                        except Exception as e_pipeline:
                            print(f"ERROR: Unhandled exception in agent pipeline for user {user_id_to_process}: {e_pipeline}")
                            import traceback; traceback.print_exc()
                else:
                    print("No new relevant emails processed or no KBs updated in this cycle.")
            else:
                print("Failed to connect to mailbox. Will retry after polling interval.")

            print(f"--- End of Polling Cycle #{run_count} ---")
            print(f"Waiting for {POLLING_INTERVAL_SECONDS} seconds before next email check...")
            time.sleep(POLLING_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Shutting down KompowLearn email poller...")
    except Exception as e_main_loop:
        print(f"FATAL ERROR in main polling loop: {e_main_loop}")
        import traceback
        traceback.print_exc()
        print("The application will exit due to this unrecoverable error.")
    finally:
        print("\n-----------------------------------------------------------")
        print(" KompowLearn - Continuous Email Polling Service Stopped. ")
        print("-----------------------------------------------------------")
