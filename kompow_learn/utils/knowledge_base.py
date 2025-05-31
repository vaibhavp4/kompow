import os
import re
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

# --- Agno Imports & Dummy Class Fallbacks ---
AGNO_AVAILABLE = False
try:
    from agno.knowledge.base import KnowledgeBase as ActualAgnoKnowledgeBase
    from agno.knowledge.document import Document as ActualAgnoDocument
    from agno.vectordb.lancedb import LanceDb as ActualAgnoLanceDb
    from agno.embedder.openai import OpenAIEmbedder as ActualAgnoOpenAIEmbedder
    AGNO_AVAILABLE = True
    print("INFO: Successfully imported Agno components from installed package.")
except ImportError as e:
    print(f"WARNING: Failed to import Agno components: {e}. KnowledgeBase functionality will use DUMMY classes. This is likely due to an issue with the 'agno' package installation or environment.")
    print("         Functionality requiring actual Agno features (like vector search, real embeddings) will be severely limited or non-operational.")

    class DummyAgnoKnowledgeBase:
        def __init__(self, vector_db=None, id=None, description=None, metadata=None):
            self.vector_db = vector_db
            self.id = id
            self.description = description
            self.metadata = metadata
            print("WARNING: Using DUMMY AgnoKnowledgeBase class.")

        def add(self, documents=None):
            print(f"DUMMY AgnoKnowledgeBase: Add called with {len(documents) if documents else 0} documents.")
            # In real Agno, add might not return a boolean directly, but raises errors on failure.
            # For testing, let's assume it indicates success if no error.
            if self.vector_db and self.vector_db.embedder is None and documents:
                 # Simulate error if trying to add to a DB that needs embeddings but has no embedder
                 # This is a specific scenario we want to test for add_document_to_kb
                 for doc in documents:
                     if doc.metadata.get("doc_type") != "flashcard_set": # Flashcard sets might be added without content embedding
                        # This is a simplification. Real LanceDB would error if a vector column exists but no vector can be made.
                        print(f"DUMMY AgnoKnowledgeBase: Mock error - cannot add document '{doc.id}' requiring embedding to DB with no embedder.")
                        # raise Exception(f"DUMMY: Cannot generate vector for document {doc.id} without an embedder.")
                        # For tests, we just log. add_document_to_kb should return False.

            return True # Or simulate what real Agno does; often no return value for success.

        def search(self, query=None, limit=1):
            print(f"DUMMY AgnoKnowledgeBase: Search called with query '{query}', limit {limit}.")
            if self.vector_db and self.vector_db.embedder is None:
                print("DUMMY AgnoKnowledgeBase: Search would fail or return nothing without an embedder.")
                return []
            # Simulate returning some dummy documents if needed for tests
            dummy_doc_content = "Dummy search result content."
            dummy_doc_meta = {"source": "dummy_search"}
            if query == "flashcard document" or (query and "flashcards by" in query): # For get_flashcard_sets_for_user test
                dummy_doc_meta = {"doc_type": "flashcard_set", "user_id":"test_user@example.com", "topic":"Dummy Topic", "creation_date": datetime.now(timezone.utc).isoformat()}
                dummy_doc_content = json.dumps([{"question": "Dummy Q", "answer": "Dummy A"}])

            return [ActualAgnoDocument(id=f"dummy_doc_{i}", content=dummy_doc_content, metadata=dummy_doc_meta) for i in range(limit)]

        def load(self): # If used by any agent, though not directly in KB code now
            print("DUMMY AgnoKnowledgeBase: Load called.")

    class DummyAgnoDocument:
        def __init__(self, content=None, metadata=None, id=None):
            self.content = content
            self.metadata = metadata or {}
            self.id = id
            # print("INFO: Using DUMMY AgnoDocument class.") # Can be too verbose

    class DummyAgnoLanceDb:
        def __init__(self, uri=None, table_name=None, embedder=None, mode=None, create_table=None):
            self.uri = uri
            self.table_name = table_name
            self.embedder = embedder # This is crucial for our existing embedder checks
            self.mode = mode
            self.create_table = create_table
            print(f"WARNING: Using DUMMY AgnoLanceDb class for table '{table_name}'. Embedder is {'SET' if embedder else 'NOT SET'}.")

    class DummyAgnoOpenAIEmbedder:
        def __init__(self, id="text-embedding-ada-002", api_key=None, client=None):
            self.id = id
            self.api_key = api_key
            self.client = client
            if not api_key or api_key == "your_openai_api_key_here":
                 print("WARNING: DUMMY AgnoOpenAIEmbedder initialized with invalid or missing API key.")
            else:
                 print("INFO: DUMMY AgnoOpenAIEmbedder initialized (simulating API key presence).")

    # Assign Dummies to the names expected by the rest of the file
    ActualAgnoKnowledgeBase = DummyAgnoKnowledgeBase
    ActualAgnoDocument = DummyAgnoDocument
    ActualAgnoLanceDb = DummyAgnoLanceDb
    ActualAgnoOpenAIEmbedder = DummyAgnoOpenAIEmbedder
# --- End of Agno Imports & Dummy Class Fallbacks ---


LANCEDB_URI_BASE = "tmp/lancedb_store"
# This will be created by get_user_knowledge_base if it doesn't exist
# os.makedirs(LANCEDB_URI_BASE, exist_ok=True) # Moved to get_user_knowledge_base

def sanitize_table_name(name: str) -> str:
    name = re.sub(r'[.@:\-/]', '_', name)
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    name = name.strip('_')
    if not name:
        return "default_table"
    return name

def get_user_knowledge_base(user_id: str) -> ActualAgnoKnowledgeBase | None:
    api_key = os.getenv("OPENAI_API_KEY")
    embedder = None
    if AGNO_AVAILABLE and api_key and api_key != "your_openai_api_key_here": # Only try real embedder if Agno and key are fine
        try:
            embedder = ActualAgnoOpenAIEmbedder(api_key=api_key)
        except Exception as e:
            print(f"Error initializing ActualAgnoOpenAIEmbedder: {e}. Proceeding without embedder.")
    elif not AGNO_AVAILABLE and api_key and api_key != "your_openai_api_key_here": # Agno not available, but API key is - use DUMMY with key
        embedder = ActualAgnoOpenAIEmbedder(api_key=api_key) # This is DummyAgnoOpenAIEmbedder if Agno failed
    # If API key is missing, embedder remains None, DummyAgnoOpenAIEmbedder won't be "successfully" init'd with key

    table_name = f"user_{sanitize_table_name(user_id)}"
    lancedb_uri = os.path.join(LANCEDB_URI_BASE, table_name)

    try:
        # Ensure the specific directory for this table exists only when creating the DB
        os.makedirs(lancedb_uri, exist_ok=True)

        vector_db = ActualAgnoLanceDb(
            uri=lancedb_uri,
            table_name=table_name,
            embedder=embedder
        )
        kb = ActualAgnoKnowledgeBase(vector_db=vector_db)
        # print(f"KnowledgeBase for user '{user_id}' (table: {table_name}) initialized. Embedder {'present' if embedder else 'absent'}.")
        return kb
    except Exception as e:
        print(f"Error creating KnowledgeBase for user {user_id} with table {table_name}: {e}")
        return None

def add_document_to_kb(kb: ActualAgnoKnowledgeBase, doc_content: str, doc_metadata: dict = None, doc_id: str = None) -> bool:
    if not kb:
        print("KB Error: KnowledgeBase instance is None. Cannot add document.")
        return False
    if not kb.vector_db.embedder:
        print(f"KB Error: Embedder not available for KB table {kb.vector_db.table_name}. Cannot add document (requires embeddings). Likely missing API key.")
        return False

    doc_metadata = doc_metadata or {}
    try:
        document = ActualAgnoDocument(content=doc_content, metadata=doc_metadata, id=doc_id)
        # The dummy add might always return True, or we can make it more complex
        # Real Agno add might not return a boolean. Let's assume success if no exception.
        kb.add(documents=[document])
        # print(f"Document '{doc_id if doc_id else 'auto_id'}' reported as added to KB table: {kb.vector_db.table_name}.")
        return True # Assuming add operation in Agno/Dummy signifies success if it doesn't raise error
    except Exception as e:
        print(f"Error adding document to KB table {kb.vector_db.table_name}: {e}")
        if "RateLimitError" in str(e): print("OpenAI Rate Limit likely exceeded.")
        return False

def query_knowledge_base(kb: ActualAgnoKnowledgeBase, query_text: str, limit: int = 3) -> list[ActualAgnoDocument] | None:
    if not kb:
        print("KB Error: KnowledgeBase instance is None. Cannot query.")
        return None
    if not kb.vector_db.embedder:
        print(f"KB Error: Embedder not available for KB table {kb.vector_db.table_name}. Cannot query (requires embeddings). Likely missing API key.")
        return [] # Return empty list for consistency, as search would yield no results
    try:
        results = kb.search(query=query_text, limit=limit)
        # print(f"Query returned {len(results) if results else 0} documents for '{query_text}'.")
        return results
    except Exception as e:
        print(f"Error querying KB table {kb.vector_db.table_name}: {e}")
        return [] # Return empty list on error

def add_flashcard_set_to_kb(kb: ActualAgnoKnowledgeBase, user_id: str, topic: str, flashcards_json_string: str, source: str = "on_demand_generation") -> bool:
    if not kb:
        print(f"KB Error: KnowledgeBase not initialized for user {user_id} (passed as None).")
        return False

    # For flashcard sets, content is JSON. If table has an embedder, it might try to embed this JSON string.
    # If embedder is None (no API key), and if LanceDB schema for some reason requires a vector, this could fail.
    # The DummyAgnoLanceDb and DummyAgnoKnowledgeBase will simulate this behavior based on embedder presence.
    if kb.vector_db.embedder is None:
         print(f"KB Warning: Embedder not available for table {kb.vector_db.table_name}. Adding flashcard set; its 'content' (JSON string) will not be semantically searchable. If table schema strictly requires vectors, this add might fail with real LanceDB.")

    timestamp = datetime.now(timezone.utc).isoformat()
    sanitized_topic_for_id = re.sub(r'[^a-zA-Z0-9_]', '_', topic.lower())[:50]
    unique_ts_part = int(datetime.now(timezone.utc).timestamp() * 1000)
    doc_id = f"flashcards_{sanitize_table_name(user_id)}_{sanitized_topic_for_id}_{unique_ts_part}"

    metadata = {"doc_type": "flashcard_set", "topic": topic, "creation_date": timestamp, "source": source, "user_id": user_id}

    try:
        parsed_flashcards = json.loads(flashcards_json_string)
        if not isinstance(parsed_flashcards, list):
            print("KB Error: Flashcards JSON string does not represent a list.")
            return False
    except json.JSONDecodeError:
        print("KB Error: Invalid JSON string provided for flashcards.")
        return False

    document = ActualAgnoDocument(id=doc_id, content=flashcards_json_string, metadata=metadata)
    try:
        kb.add(documents=[document])
        print(f"Flashcard set '{doc_id}' (topic: {topic}) reported as added to KB table: {kb.vector_db.table_name}.")
        return True
    except Exception as e:
        print(f"Error during kb.add for flashcard set '{doc_id}' in table {kb.vector_db.table_name}: {e}")
        return False

def get_flashcard_sets_for_user(kb: ActualAgnoKnowledgeBase, user_id: str, topic: str = None, limit: int = 20) -> list[ActualAgnoDocument]:
    if not kb:
        print(f"KB Error: KnowledgeBase not initialized for user {user_id} (passed as None).")
        return []
    if not kb.vector_db.embedder:
        print(f"KB Error: Embedder not available for KB. Cannot use semantic search for flashcard sets. Please set OPENAI_API_KEY.")
        return []

    query_text = f"flashcards by {user_id} about {topic}" if topic else f"flashcard sets related to user {user_id}"
    try:
        search_results = kb.search(query=query_text, limit=limit * 10) # Fetch more to filter
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
    return flashcard_docs[:limit]

def get_available_flashcard_topics(kb: ActualAgnoKnowledgeBase, user_id: str) -> list[str]:
    if not kb:
        print(f"KB Error: KnowledgeBase not initialized for user {user_id} (passed as None) for topic retrieval.")
        return []
    if not kb.vector_db.embedder:
        print(f"KB Error: Embedder not available for KB. Cannot list flashcard topics (requires search). Please set OPENAI_API_KEY.")
        return []

    all_flashcard_docs = get_flashcard_sets_for_user(kb, user_id, limit=1000)
    topics = sorted(list(set(
        doc.metadata.get("topic") for doc in all_flashcard_docs if doc.metadata and doc.metadata.get("topic")
    )))
    return topics

if __name__ == "__main__":
    # This __main__ block is primarily for testing the KB functionalities with DUMMY or REAL Agno.
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env') # Assumes utils is one level down
    load_dotenv(dotenv_path=dotenv_path)
    print(f"--- knowledge_base.py __main__ Test ---")
    print(f"AGNO_AVAILABLE: {AGNO_AVAILABLE}")

    api_key_present_for_test = os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here"
    print(f"OpenAI API Key is considered present for test: {api_key_present_for_test}")

    test_user = "kb_main_test@example.com"
    kb_instance_main = get_user_knowledge_base(test_user)

    if kb_instance_main:
        print(f"KB instance for '{test_user}' obtained. Embedder on KB: {'Present' if kb_instance_main.vector_db.embedder else 'Absent'}")

        # Test adding a regular document
        if kb_instance_main.vector_db.embedder: # This check is now crucial
            print("\nTesting add_document_to_kb (requires embedder)...")
            add_doc_success = add_document_to_kb(kb_instance_main, "Test content for regular doc.", {"type":"general"}, "test_doc_01")
            print(f"add_document_to_kb success: {add_doc_success}")
            if add_doc_success :
                results = query_knowledge_base(kb_instance_main, "Test content", limit=1)
                print(f"Query results for 'Test content': {results[0].id if results else 'None'}")
        else:
            print("\nSkipping add_document_to_kb and query_knowledge_base tests as embedder is not available (likely no API key).")

        # Test flashcard operations
        print("\nTesting flashcard operations...")
        fc_json = json.dumps([{"q":"Test Q1","a":"Test A1"}])
        fc_add_success = add_flashcard_set_to_kb(kb_instance_main, test_user, "Test Topic Alpha", fc_json)
        print(f"add_flashcard_set_to_kb success: {fc_add_success}")

        fc_json_2 = json.dumps([{"q":"Test Q2","a":"Test A2"}])
        add_flashcard_set_to_kb(kb_instance_main, test_user, "Test Topic Beta", fc_json_2)


        if kb_instance_main.vector_db.embedder: # get_flashcard_sets and get_topics need embedder for search
            print("\nTesting get_flashcard_sets_for_user (requires embedder)...")
            sets = get_flashcard_sets_for_user(kb_instance_main, test_user, topic="Test Topic Alpha")
            print(f"Retrieved sets for 'Test Topic Alpha': {len(sets)}")
            if sets: print(f"First set content: {sets[0].content}, metadata: {sets[0].metadata}")

            print("\nTesting get_available_flashcard_topics (requires embedder)...")
            topics = get_available_flashcard_topics(kb_instance_main, test_user)
            print(f"Available topics for '{test_user}': {topics}")
        else:
            print("\nSkipping get_flashcard_sets_for_user and get_available_flashcard_topics as embedder is not available.")
            print("Note: Flashcard sets may have been added without embedding, but cannot be retrieved by current search-based methods without an embedder.")

    else:
        print(f"Failed to get KB instance for '{test_user}'.")
    print("--- knowledge_base.py __main__ Test Complete ---")
