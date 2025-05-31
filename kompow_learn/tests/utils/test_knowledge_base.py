import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
import json
from datetime import datetime, timezone

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# utils.knowledge_base imports ActualAgno... classes which are dummies if agno is not found.
from utils.knowledge_base import (
    get_user_knowledge_base,
    add_document_to_kb,
    query_knowledge_base,
    add_flashcard_set_to_kb,
    get_flashcard_sets_for_user,
    get_available_flashcard_topics,
    ActualAgnoKnowledgeBase, # Used to check instance types, will be Dummy if Agno failed
    ActualAgnoDocument,
    ActualAgnoLanceDb,
    ActualAgnoOpenAIEmbedder,
    AGNO_AVAILABLE # To know if we are testing with real or dummy Agno
)

class TestKnowledgeBase(unittest.TestCase):

    def setUp(self):
        # Patch os.makedirs to prevent actual directory creation during tests
        self.makedirs_patcher = patch('os.makedirs')
        self.mock_makedirs = self.makedirs_patcher.start()

        # Patch load_dotenv to do nothing as we'll mock os.getenv
        self.dotenv_patcher = patch('dotenv.load_dotenv')
        self.mock_dotenv = self.dotenv_patcher.start()

        # Store original AGNO_AVAILABLE and potentially mock it if needed for specific tests
        self.original_agno_available = AGNO_AVAILABLE
        # If we need to force a test for when Agno is available vs not, we can patch
        # `utils.knowledge_base.AGNO_AVAILABLE` here.

    def tearDown(self):
        self.makedirs_patcher.stop()
        self.dotenv_patcher.stop()
        # Restore original AGNO_AVAILABLE if changed
        # utils.knowledge_base.AGNO_AVAILABLE = self.original_agno_available


    @patch('utils.knowledge_base.os.getenv')
    def test_get_user_knowledge_base_api_key_present(self, mock_getenv):
        mock_getenv.return_value = "fake_openai_api_key" # Simulate API key is present

        kb = get_user_knowledge_base("test_user_api_present")
        self.assertIsNotNone(kb)
        self.assertIsInstance(kb, ActualAgnoKnowledgeBase)
        self.assertIsNotNone(kb.vector_db)
        self.assertIsInstance(kb.vector_db, ActualAgnoLanceDb)
        self.assertIsNotNone(kb.vector_db.embedder)
        self.assertIsInstance(kb.vector_db.embedder, ActualAgnoOpenAIEmbedder)
        self.mock_makedirs.assert_called() # Check if directory creation was attempted

    @patch('utils.knowledge_base.os.getenv')
    def test_get_user_knowledge_base_api_key_missing(self, mock_getenv):
        mock_getenv.return_value = None # Simulate API key is missing

        kb = get_user_knowledge_base("test_user_api_missing")
        self.assertIsNotNone(kb) # KB object should still be returned (with dummy classes if Agno unavailable)
        self.assertIsInstance(kb, ActualAgnoKnowledgeBase)
        self.assertIsNotNone(kb.vector_db)
        self.assertIsInstance(kb.vector_db, ActualAgnoLanceDb)
        # Crucially, the embedder should be None if API key is missing
        self.assertIsNone(kb.vector_db.embedder, "Embedder should be None when API key is missing.")

    @patch('utils.knowledge_base.os.getenv')
    def test_add_document_to_kb_success_with_embedder(self, mock_getenv):
        mock_getenv.return_value = "fake_openai_api_key" # Ensure embedder is attempted

        # Mock the KnowledgeBase's add method that's part of the dummy or real Agno
        with patch.object(ActualAgnoKnowledgeBase, 'add', return_value=True) as mock_kb_add:
            kb = get_user_knowledge_base("test_user_add_doc")
            self.assertIsNotNone(kb.vector_db.embedder, "Test requires embedder to be present for this scenario.")

            success = add_document_to_kb(kb, "Test content", {"source": "test"}, "doc1")
            self.assertTrue(success)
            mock_kb_add.assert_called_once()
            # Check that an ActualAgnoDocument was passed to kb.add
            args, kwargs = mock_kb_add.call_args
            self.assertTrue('documents' in kwargs)
            self.assertEqual(len(kwargs['documents']), 1)
            self.assertIsInstance(kwargs['documents'][0], ActualAgnoDocument)
            self.assertEqual(kwargs['documents'][0].id, "doc1")

    @patch('utils.knowledge_base.os.getenv')
    def test_add_document_to_kb_no_embedder_fails(self, mock_getenv):
        mock_getenv.return_value = None # API key missing, so no embedder

        with patch.object(ActualAgnoKnowledgeBase, 'add') as mock_kb_add:
            kb = get_user_knowledge_base("test_user_add_doc_no_embed")
            self.assertIsNone(kb.vector_db.embedder, "Embedder must be None for this test.")

            success = add_document_to_kb(kb, "Test content", {"source": "test"}, "doc1")
            self.assertFalse(success, "add_document_to_kb should fail if embedder is required but missing.")
            mock_kb_add.assert_not_called() # kb.add should not be called if embedder check fails

    @patch('utils.knowledge_base.os.getenv')
    def test_add_flashcard_set_to_kb_valid_json(self, mock_getenv):
        mock_getenv.return_value = "fake_api_key_for_flashcard_add" # Embedder might be present

        with patch.object(ActualAgnoKnowledgeBase, 'add', return_value=True) as mock_kb_add:
            kb = get_user_knowledge_base("test_user_fc_add")

            fc_list = [{"q": "Q1", "a": "A1"}]
            fc_json_str = json.dumps(fc_list)
            success = add_flashcard_set_to_kb(kb, "test_user_fc_add", "Test Topic", fc_json_str)

            self.assertTrue(success)
            mock_kb_add.assert_called_once()
            args, kwargs = mock_kb_add.call_args
            doc_arg = kwargs['documents'][0]
            self.assertIsInstance(doc_arg, ActualAgnoDocument)
            self.assertEqual(doc_arg.content, fc_json_str)
            self.assertEqual(doc_arg.metadata.get("doc_type"), "flashcard_set")
            self.assertEqual(doc_arg.metadata.get("topic"), "Test Topic")

    @patch('utils.knowledge_base.os.getenv')
    def test_add_flashcard_set_to_kb_invalid_json_string(self, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        with patch.object(ActualAgnoKnowledgeBase, 'add') as mock_kb_add:
            kb = get_user_knowledge_base("test_user_fc_invalid")
            invalid_json_str = "not a json list"
            success = add_flashcard_set_to_kb(kb, "test_user_fc_invalid", "Invalid Topic", invalid_json_str)
            self.assertFalse(success)
            mock_kb_add.assert_not_called()

    @patch('utils.knowledge_base.os.getenv')
    def test_add_flashcard_set_to_kb_json_not_a_list(self, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        with patch.object(ActualAgnoKnowledgeBase, 'add') as mock_kb_add:
            kb = get_user_knowledge_base("test_user_fc_not_list")
            valid_json_but_not_list = json.dumps({"q": "Q1", "a": "A1"}) # a dict, not a list
            success = add_flashcard_set_to_kb(kb, "test_user_fc_not_list", "Not List Topic", valid_json_but_not_list)
            self.assertFalse(success)
            mock_kb_add.assert_not_called()

    @patch('utils.knowledge_base.os.getenv')
    def test_get_flashcard_sets_no_embedder(self, mock_getenv):
        mock_getenv.return_value = None # No API key -> no embedder
        kb = get_user_knowledge_base("test_user_get_fc_no_embed")
        self.assertIsNone(kb.vector_db.embedder, "Embedder should be None for this test.")

        # Mock kb.search directly on the instance because the dummy class's search might behave differently
        # For this test, the function should return early due to no embedder.
        # If AGNO_AVAILABLE is False, kb.search is already a dummy that considers embedder.
        # If AGNO_AVAILABLE is True, we'd be testing real Agno's behavior with no embedder, which should also error or return [].

        # If not AGNO_AVAILABLE, the dummy search will print "Search would fail or return nothing" and return []
        # If AGNO_AVAILABLE, real Agno search would likely error if embedder is None and it's needed.
        # The function get_flashcard_sets_for_user itself checks for embedder.

        results = get_flashcard_sets_for_user(kb, "test_user_get_fc_no_embed", "Any Topic")
        self.assertEqual(results, [])

    @patch('utils.knowledge_base.os.getenv')
    def test_get_flashcard_sets_with_results_and_filtering(self, mock_getenv):
        mock_getenv.return_value = "fake_api_key" # Embedder will be present

        user_id = "user_fc_results"
        topic_filter = "Target Topic"

        mock_doc1_content = json.dumps([{"q":"Q1_T1","a":"A1_T1"}])
        mock_doc1 = ActualAgnoDocument(id="fc_doc1", content=mock_doc1_content, metadata={
            "doc_type": "flashcard_set", "user_id": user_id, "topic": topic_filter, "creation_date": "2023-01-01T12:00:00Z"
        })
        mock_doc2_content = json.dumps([{"q":"Q1_T2","a":"A1_T2"}])
        mock_doc2 = ActualAgnoDocument(id="fc_doc2", content=mock_doc2_content, metadata={
            "doc_type": "flashcard_set", "user_id": user_id, "topic": "Other Topic", "creation_date": "2023-01-02T12:00:00Z"
        })
        mock_doc3_content = json.dumps([{"q":"Q2_T1","a":"A2_T1"}]) # Another for target topic
        mock_doc3 = ActualAgnoDocument(id="fc_doc3", content=mock_doc3_content, metadata={
            "doc_type": "flashcard_set", "user_id": user_id, "topic": topic_filter, "creation_date": "2023-01-03T12:00:00Z"
        })
        mock_doc4_general = ActualAgnoDocument(id="general_doc", content="some text", metadata={"user_id": user_id, "doc_type": "general"})

        # Mock the search method of the KnowledgeBase instance
        # The dummy search might return some generic things, so direct mocking is better for this test.
        mock_search_results = [mock_doc1, mock_doc2, mock_doc3, mock_doc4_general]

        with patch.object(ActualAgnoKnowledgeBase, 'search', return_value=mock_search_results) as mock_kb_search:
            kb = get_user_knowledge_base(user_id)
            self.assertIsNotNone(kb.vector_db.embedder, "Embedder should be present for this test.")

            # Test with topic filter
            results = get_flashcard_sets_for_user(kb, user_id, topic=topic_filter, limit=5)
            mock_kb_search.assert_called_once() # Or more depending on internal logic of get_flashcard_sets

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].id, "fc_doc3") # Sorted by date descending
            self.assertEqual(results[1].id, "fc_doc1")
            self.assertTrue(all(doc.metadata.get("topic") == topic_filter for doc in results))

            # Test without topic filter (should get all for user_id)
            mock_kb_search.reset_mock() # Reset for next call
            # If dummy search is too simple, provide specific mock results for the no-topic case too.
            # The dummy currently returns generic flashcards if "flashcards by" is in query.
            # Let's refine the mock search_results for the no-topic case if needed,
            # or ensure the dummy search is good enough.
            # For this test, let's assume the same mock_search_results are fine for broad query.

            results_no_topic = get_flashcard_sets_for_user(kb, user_id, limit=5)
            self.assertEqual(len(results_no_topic), 3) # doc1, doc2, doc3 are flashcard_sets for this user
            self.assertEqual(results_no_topic[0].id, "fc_doc3") # still sorted by date
            self.assertEqual(results_no_topic[1].id, "fc_doc2")
            self.assertEqual(results_no_topic[2].id, "fc_doc1")


    @patch('utils.knowledge_base.os.getenv')
    def test_get_available_flashcard_topics(self, mock_getenv):
        mock_getenv.return_value = "fake_api_key" # Embedder present
        user_id = "user_topics_test"

        mock_doc_topic_A_1 = ActualAgnoDocument(id="fc_A1", content="[]", metadata={"doc_type": "flashcard_set", "user_id": user_id, "topic": "Topic A"})
        mock_doc_topic_B_1 = ActualAgnoDocument(id="fc_B1", content="[]", metadata={"doc_type": "flashcard_set", "user_id": user_id, "topic": "Topic B"})
        mock_doc_topic_A_2 = ActualAgnoDocument(id="fc_A2", content="[]", metadata={"doc_type": "flashcard_set", "user_id": user_id, "topic": "Topic A"}) # Duplicate topic
        mock_doc_no_topic = ActualAgnoDocument(id="fc_no_topic", content="[]", metadata={"doc_type": "flashcard_set", "user_id": user_id}) # No topic metadata
        mock_general_doc = ActualAgnoDocument(id="general1", content="text", metadata={"doc_type": "general", "user_id": user_id})

        # Mock the search that get_flashcard_sets_for_user (called by get_available_flashcard_topics) uses
        mock_search_results_for_topics = [mock_doc_topic_A_1, mock_doc_topic_B_1, mock_doc_topic_A_2, mock_doc_no_topic, mock_general_doc]

        with patch.object(ActualAgnoKnowledgeBase, 'search', return_value=mock_search_results_for_topics) as mock_kb_search_topics:
            kb = get_user_knowledge_base(user_id)
            self.assertIsNotNone(kb.vector_db.embedder)

            topics = get_available_flashcard_topics(kb, user_id)
            # get_flashcard_sets_for_user is called by get_available_flashcard_topics, so search will be called.
            mock_kb_search_topics.assert_called()

            self.assertEqual(topics, sorted(["Topic A", "Topic B"]))

if __name__ == '__main__':
    unittest.main()
