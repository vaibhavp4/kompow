# KompowLearn Application: Deployment and Local Running Guide

This document provides instructions on how to set up and run the KompowLearn application locally. KompowLearn consists of several key components:
- An email processing service (`main.py`) that polls an email account, processes content, and uses AI agents to generate flashcards.
- A Gradio web interface (`ui/app.py`) for interacting with the system (generating flashcards on-demand, viewing stored flashcards).
- A FastAPI server (`api_server.py`) exposing endpoints for external integrations (e.g., for a WhatsApp bot).

## 1. Prerequisites

*   **Python:** Python 3.10 or higher is recommended.
*   **pip:** Python package installer.
*   **Virtual Environment (Recommended):** `venv` or `conda`.
*   **Git:** For cloning the repository.
*   **OpenAI API Key:** Required for AI agent functionalities (text embedding, LLM calls for analysis, research, and flashcard generation).
*   **IMAP Email Account:** A dedicated email account (e.g., `learn@kompow.com`) for the system to poll. Ensure IMAP access is enabled.

## 2. Setup Instructions

### 2.1. Clone the Repository
   ```bash
   git clone <repository_url>
   cd kompowlearn # Or your repository's root directory
   ```

### 2.2. Create and Activate a Virtual Environment
   **Using `venv`:**
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

### 2.3. Install Dependencies
   Install all required Python packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

### 2.4. Configure Environment Variables
   Create a `.env` file in the root directory of the project. This file will store sensitive credentials and configuration. Copy the example below and replace placeholder values with your actual credentials.

   **`.env` file contents:**
   ```env
   # OpenAI API Key (Required for all AI functionalities)
   OPENAI_API_KEY="your_openai_api_key_here"

   # Email Account Credentials (Required for main.py email polling)
   EMAIL_HOST="your_imap_server.com" # e.g., imap.gmail.com
   EMAIL_USER="your_email@example.com"
   EMAIL_PASS="your_email_password_or_app_password"

   # Email Polling Configuration (Optional, defaults will be used if not set)
   # POLLING_INTERVAL_SECONDS=300 # Interval in seconds for checking new emails (default: 300)
   # MAX_EMAILS_PER_CYCLE=10      # Max emails to process in one polling cycle (default: 10)

   # LanceDB URI (Optional, defaults to tmp/lancedb_store in project root)
   # LANCEDB_URI_BASE="tmp/lancedb_store"
   ```

   **Important Notes on Environment Variables:**
   *   **`OPENAI_API_KEY`:** This is essential. Without a valid key, AI agents, embedding generation, and core functionalities will fail.
   *   **`EMAIL_HOST`, `EMAIL_USER`, `EMAIL_PASS`:** These are required to run the `main.py` email polling service. For Gmail, you might need to enable "Less secure app access" or generate an "App Password".
   *   Ensure the `.env` file is never committed to version control if it contains real credentials. Add `.env` to your `.gitignore` file.

## 3. Running the Application Components

The application has three main runnable components. You can run them in separate terminal windows. Ensure your virtual environment is activated for each.

### 3.1. Running the FastAPI Server (`api_server.py`)
   This server exposes API endpoints for external integrations.
   ```bash
   python api_server.py
   ```
   By default, it runs on `http://0.0.0.0:8000`.
   You can access the API documentation (Swagger UI) at `http://localhost:8000/docs`.

   **Note:** The API server will perform critical checks for `OPENAI_API_KEY` on startup. If the key is invalid or missing, it will print an error and exit.

### 3.2. Running the Gradio Web Interface (`ui/app.py`)
   This provides a user interface for generating and viewing flashcards.
   ```bash
   python ui/app.py
   ```
   The Gradio app will typically be available at a local URL like `http://127.0.0.1:7860` (the exact URL will be shown in the console).

   **Note:** The Gradio UI also depends on `OPENAI_API_KEY` for its functionalities. It will display error messages if the key is not configured correctly.

### 3.3. Running the Email Polling and Processing Service (`main.py`)
   This service continuously polls the configured email account, processes new emails, updates user knowledge bases, and triggers the AI agent pipeline to generate flashcards.
   ```bash
   python main.py
   ```
   **Prerequisites for `main.py`:**
   *   Ensure `OPENAI_API_KEY`, `EMAIL_HOST`, `EMAIL_USER`, and `EMAIL_PASS` are correctly set in your `.env` file.
   *   The IMAP email account must be accessible.

   The service will log its activities to the console. To stop it, press `Ctrl+C`.

## 4. Application Workflow Overview

1.  **Email Processing (`main.py`):**
    *   Polls the IMAP account specified in `.env`.
    *   For new emails, content (body, text from attachments like TXT/PDF/DOCX, crawled web links) is extracted.
    *   This content is added to a user-specific knowledge base (LanceDB store). The user is typically identified by their email address (sender of the email).
2.  **AI Agent Pipeline (Triggered by `main.py` for users with new content):**
    *   `LearningProfileAgent`: Analyzes the user's updated knowledge base to determine learning interests/topics.
    *   `ResearchAgent`: Researches these topics using web search.
    *   `FlashcardGenerationAgent`: Creates flashcards from the researched content.
    *   The newly generated flashcards are then stored back into the user's knowledge base.
3.  **Gradio UI (`ui/app.py`):**
    *   Allows users to manually generate flashcards for any topic. These are stored under a default "on_demand_user" profile.
    *   Allows users to view stored flashcards (currently for the "on_demand_user").
4.  **FastAPI Server (`api_server.py`):**
    *   Exposes endpoints to:
        *   Generate flashcards for a topic (similar to Gradio's on-demand feature, stored for "api_on_demand_user").
        *   Retrieve stored flashcards for a specified `user_id` and optional topic.
        *   List available flashcard topics for a `user_id`.
    *   This API can be used by external services (e.g., a WhatsApp bot built with n8n or Botpress) to integrate with KompowLearn.

## 5. Missing Utilities / Known Limitations / Future Considerations

*   **User Management:** Currently, user identification is basic (sender's email for `main.py`, default users for UI/API). A proper user authentication and management system is not implemented.
*   **Knowledge Base Topic Retrieval:** The current method in `utils.knowledge_base.get_flashcard_sets_for_user` and `get_available_flashcard_topics` relies on semantic search (requiring a functional embedder/API key) to fetch documents broadly before filtering by metadata. This can be inefficient. A more direct metadata filtering capability in LanceDB (if exposed appropriately through Agno's `KnowledgeBase` or by directly using the LanceDB table object) would be more performant, especially for listing topics or retrieving flashcards when the embedder is unavailable.
*   **Error Handling:** While error handling is implemented, it can be further enhanced for production scenarios (e.g., more granular logging, retry mechanisms for external calls).
*   **Scalability:** The current setup is designed for local execution. Scaling for multiple users or high load would require further architectural considerations (e.g., task queues, database optimization, stateless API workers).
*   **Cost Management:** Extensive use of OpenAI models will incur costs. Monitor API usage.
*   **Security:** For a production deployment, security aspects like API authentication, input validation, and protection against prompt injection would need careful review and enhancement.
*   **Frontend for Email Users:** The Gradio UI currently primarily serves the "on_demand_user". To allow users whose emails are processed by `main.py` to see *their* flashcards, the Gradio UI would need to be extended to support user selection or authentication, and then use the selected `user_id` when calling knowledge base functions.

This guide should help you get KompowLearn up and running locally.
```
