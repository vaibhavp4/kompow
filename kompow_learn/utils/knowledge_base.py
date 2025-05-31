from agno.knowledge.base import KnowledgeBase
from agno.knowledge.document import Document
from agno.vectordb.lancedb import LanceDb
from agno.embedder.openai import OpenAIEmbedder
import os
import re
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

LANCEDB_URI_BASE = "tmp/lancedb_store"
os.makedirs(LANCEDB_URI_BASE, exist_ok=True)

def sanitize_table_name(name: str) -> str:
    name = re.sub(r'[.@:\-/]', '_', name)
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    name = name.strip('_')
    if not name:
        return "default_table"
    return name

def get_user_knowledge_base(user_id: str) -> KnowledgeBase | None:
    api_key = os.getenv("OPENAI_API_KEY")
    embedder = None
    if api_key and api_key != "your_openai_api_key_here":
        try:
            embedder = OpenAIEmbedder(api_key=api_key)
            # print("OpenAIEmbedder initialized for get_user_knowledge_base.") # Less verbose
        except Exception as e:
            print(f"Error initializing OpenAIEmbedder: {e}. Proceeding without embedder for KB structure operations.")
    # else: # Less verbose, handled by embedder checks in functions needing it
        # print("OPENAI_API_KEY not found or is a placeholder. KB operations requiring embeddings will fail.")

    table_name = f"user_{sanitize_table_name(user_id)}"
    lancedb_uri = os.path.join(LANCEDB_URI_BASE, table_name)

    try:
        os.makedirs(lancedb_uri, exist_ok=True)
        vector_db = LanceDb(
            uri=lancedb_uri,
            table_name=table_name,
            embedder=embedder
        )
        # print(f"LanceDb instance created for table '{table_name}'. Embedder is {'SET' if embedder else 'NOT SET'}.")
        kb = KnowledgeBase(vector_db=vector_db)
        # print(f"KnowledgeBase for user '{user_id}' (table: {table_name}) initialized.")
        return kb
    except Exception as e:
        print(f"Error creating KnowledgeBase for user {user_id} with table {table_name}: {e}")
        return None

def add_document_to_kb(kb: KnowledgeBase, doc_content: str, doc_metadata: dict = None, doc_id: str = None) -> bool:
    if not kb:
        print("KnowledgeBase instance is None. Cannot add document.")
        return False
    if not kb.vector_db.embedder:
        print(f"Error: Embedder not available for KB table {kb.vector_db.table_name}. Cannot add document (requires embeddings). Likely missing API key.")
        return False

    doc_metadata = doc_metadata or {}
    try:
        document = Document(content=doc_content, metadata=doc_metadata, id=doc_id)
        kb.add(documents=[document])
        # print(f"Document '{doc_id if doc_id else 'with_content_hash'}' added to KB table: {kb.vector_db.table_name}.")
        return True
    except Exception as e:
        print(f"Error adding document to KB table {kb.vector_db.table_name}: {e}")
        if "RateLimitError" in str(e): print("OpenAI Rate Limit likely exceeded.")
        return False

def query_knowledge_base(kb: KnowledgeBase, query_text: str, limit: int = 3) -> list[Document] | None:
    if not kb:
        print("KnowledgeBase instance is None. Cannot query.")
        return None
    if not kb.vector_db.embedder:
        print(f"Error: Embedder not available for KB table {kb.vector_db.table_name}. Cannot query (requires embeddings). Likely missing API key.")
        return None
    try:
        results = kb.search(query=query_text, limit=limit)
        # print(f"Found {len(results)} documents for query '{query_text}' in table {kb.vector_db.table_name}.")
        return results
    except Exception as e:
        print(f"Error querying KnowledgeBase table {kb.vector_db.table_name}: {e}")
        return None

def add_flashcard_set_to_kb(kb: KnowledgeBase, user_id: str, topic: str, flashcards_json_string: str, source: str = "on_demand_generation") -> bool:
    if not kb:
        print(f"Error: KnowledgeBase not initialized for user {user_id} (passed as None).")
        return False

    # As discussed, adding flashcard sets (metadata-heavy JSON strings) might not strictly need embedding
    # *if* we only retrieve them by metadata. However, if the table has an embedder configured,
    # LanceDB/Agno will likely try to embed the 'content' field.
    # If the embedder is None (e.g., no API key), this add operation might fail if the LanceDB table schema
    # expects a vector and cannot generate one. This behavior is a bit nuanced.
    # For now, we proceed, and errors during kb.add() will be caught.
    if kb.vector_db.embedder is None:
        print(f"Warning: Embedder not available for KB table {kb.vector_db.table_name}. Adding flashcard set. If the table schema expects vectors, this might fail. Semantic search on content will not work.")


    timestamp = datetime.now(timezone.utc).isoformat()
    sanitized_topic_for_id = re.sub(r'[^a-zA-Z0-9_]', '_', topic.lower())[:50]
    unique_ts_part = int(datetime.now(timezone.utc).timestamp() * 1000)
    doc_id = f"flashcards_{sanitize_table_name(user_id)}_{sanitized_topic_for_id}_{unique_ts_part}"

    metadata = {
        "doc_type": "flashcard_set", "topic": topic, "creation_date": timestamp,
        "source": source, "user_id": user_id
    }

    try:
        parsed_flashcards = json.loads(flashcards_json_string)
        if not isinstance(parsed_flashcards, list):
            print("Error adding flashcard set: Flashcards JSON string does not represent a list.")
            return False
    except json.JSONDecodeError:
        print("Error adding flashcard set: Invalid JSON string provided for flashcards.")
        return False

    document = Document(id=doc_id, content=flashcards_json_string, metadata=metadata)

    # print(f"Attempting to add flashcard set to KB for user {user_id}, topic '{topic}', doc_id: {doc_id}")
    try:
        kb.add(documents=[document])
        print(f"Flashcard set '{doc_id}' added successfully to table {kb.vector_db.table_name}.")
        return True
    except Exception as e:
        print(f"Error during kb.add for flashcard set '{doc_id}' in table {kb.vector_db.table_name}: {e}")
        if "embedder" in str(e).lower() or "api key" in str(e).lower() or "vector" in str(e).lower():
             print("This error might be due to issues with the embedder (e.g. missing API key) or table schema expecting a vector that couldn't be generated.")
        return False

def get_flashcard_sets_for_user(kb: KnowledgeBase, user_id: str, topic: str = None, limit: int = 20) -> list[Document]:
    if not kb:
        print(f"Error: KnowledgeBase not initialized for user {user_id} (passed as None).")
        return []

    # This function relies on semantic search to find documents that are *likely* flashcard sets,
    # then filters by metadata. This is a workaround if direct metadata-only queries are not
    # easily exposed by Agno's KnowledgeBase abstraction for LanceDB.
    # This means an embedder IS REQUIRED for this function to work as implemented.
    if not kb.vector_db.embedder:
        print(f"Error: Embedder not available for KB table {kb.vector_db.table_name}. Cannot use semantic search to find flashcard sets. Please set OPENAI_API_KEY.")
        return []

    query_text = f"flashcards by {user_id} about {topic}" if topic else f"flashcard sets related to user {user_id}"
    # print(f"Performing semantic search for flashcard sets with query: '{query_text}' then filtering...")
    try:
        # Fetch more results than limit to allow for effective filtering.
        # The multiplier (e.g., *10) depends on how many non-flashcard docs might match the broad query.
        search_results = kb.search(query=query_text, limit=limit * 10)
    except Exception as e:
        print(f"Error during semantic search for flashcard sets: {e}")
        return []

    flashcard_docs = []
    if search_results:
        for doc in search_results:
            meta = doc.metadata or {}
            if meta.get("doc_type") == "flashcard_set" and meta.get("user_id") == user_id:
                if topic:
                    if meta.get("topic") == topic:
                        flashcard_docs.append(doc)
                else:
                    flashcard_docs.append(doc)

    flashcard_docs.sort(key=lambda d: (d.metadata or {}).get("creation_date", ""), reverse=True)
    # print(f"Found {len(flashcard_docs)} flashcard sets after filtering for user '{user_id}' (topic: {topic if topic else 'any'}).")
    return flashcard_docs[:limit]

def get_available_flashcard_topics(kb: KnowledgeBase, user_id: str) -> list[str]:
    """Retrieves a list of unique topics for which flashcard sets exist for a user."""
    if not kb:
        print(f"Error: KnowledgeBase not initialized for user {user_id} (passed as None) for topic retrieval.")
        return []

    # This also relies on get_flashcard_sets_for_user, which uses semantic search.
    # So, an embedder is required here too.
    if not kb.vector_db.embedder:
        print(f"Error: Embedder not available for KB. Cannot list flashcard topics (requires search). Please set OPENAI_API_KEY.")
        return []

    print(f"Fetching all flashcard sets for user '{user_id}' to extract topics...")
    # Fetch a large number of sets to get comprehensive topic list.
    all_flashcard_docs = get_flashcard_sets_for_user(kb, user_id, limit=1000)

    topics = sorted(list(set(
        doc.metadata.get("topic")
        for doc in all_flashcard_docs
        if doc.metadata and doc.metadata.get("topic")
    )))
    print(f"Found {len(topics)} unique flashcard topics for user '{user_id}': {topics}")
    return topics


if __name__ == "__main__":
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Attempting to load OPENAI_API_KEY from: {dotenv_path}")
    api_key_present = os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here"
    print(f"OPENAI_API_KEY is set for KB tests: {api_key_present}")

    test_user_id_kb = "kb_topics_test_user@example.com"
    print(f"\n--- Testing KnowledgeBase for user: {test_user_id_kb} ---")

    kb_instance = get_user_knowledge_base(test_user_id_kb)

    if kb_instance:
        print(f"KB instance for table: {kb_instance.vector_db.table_name}")

        # Test adding flashcard sets for different topics
        topics_to_add = ["General Science", "World History", "Python Basics"]
        all_added_successfully = True
        if api_key_present: # Adding requires embedding if embedder is set from API key
            for t_idx, t in enumerate(topics_to_add):
                fc_list_ex = [{"q": f"Q{t_idx+1} for {t}", "a": f"A{t_idx+1} for {t}"}]
                fc_json_ex = json.dumps(fc_list_ex)
                print(f"Adding flashcard set for topic: '{t}'")
                success = add_flashcard_set_to_kb(kb_instance, test_user_id_kb, t, fc_json_ex, source="kb_test_main")
                if not success: all_added_successfully = False
            print(f"Initial flashcard sets added (overall success: {all_added_successfully})")
        else:
            # Try adding one flashcard set - this might fail if table expects vectors and can't generate them
            print("OPENAI_API_KEY not set. Attempting to add one flashcard set (may fail if table expects vectors)...")
            fc_list_ex = [{"q": "Q1 for NoKeyTopic", "a": "A1 for NoKeyTopic"}]
            fc_json_ex = json.dumps(fc_list_ex)
            add_success_no_key = add_flashcard_set_to_kb(kb_instance, test_user_id_kb, "NoKeyTopic", fc_json_ex)
            print(f"Adding flashcard set without API key was successful: {add_success_no_key} (Note: This might mean it stored without vector, or failed if vector is required by schema).")


        # Test retrieving available topics
        print("\n--- Testing Get Available Flashcard Topics ---")
        if not api_key_present and not kb_instance.vector_db.embedder : # Added a check here for clarity for this test
            print("Skipping get_available_flashcard_topics test as it requires an embedder (API key) for its current implementation.")
        else:
            available_topics = get_available_flashcard_topics(kb_instance, test_user_id_kb)
            if available_topics:
                print(f"Available flashcard topics for user '{test_user_id_kb}': {available_topics}")

                # Test retrieving flashcards for the first available topic
                if available_topics:
                    first_topic = available_topics[0]
                    print(f"\nRetrieving flashcards for first topic: '{first_topic}'")
                    retrieved_sets = get_flashcard_sets_for_user(kb_instance, test_user_id_kb, topic=first_topic, limit=2)
                    if retrieved_sets:
                        print(f"Found {len(retrieved_sets)} set(s) for topic '{first_topic}'. Content of first set:")
                        try:
                            print(json.loads(retrieved_sets[0].content))
                        except: print("Could not parse content of first set.")
                    else:
                        print(f"No sets found for topic '{first_topic}' after adding them. Check logic or data.")
            else:
                print(f"No topics found. This is expected if adding failed or no flashcard sets of doc_type 'flashcard_set' exist with topics.")

    else:
        print(f"Failed to initialize KnowledgeBase for user '{test_user_id_kb}'. Cannot run tests.")
    print("\n--- KnowledgeBase Topic/Flashcard Retrieval Test Complete ---")
