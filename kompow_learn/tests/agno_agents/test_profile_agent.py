import unittest
from unittest.mock import patch, MagicMock, PropertyMock, ANY
import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agno_agents.profile_agent import LearningProfileAgent
from utils.knowledge_base import AGNO_AVAILABLE

if AGNO_AVAILABLE:
    from agno.knowledge.document import Document as ActualUtilDocument
    print("TestProfileAgent: Using REAL AgnoDocument for mock data.")
else:
    from utils.knowledge_base import ActualAgnoDocument as ActualUtilDocument
    print("TestProfileAgent: Using DUMMY AgnoDocument for mock data due to Agno import issues.")


class TestLearningProfileAgent(unittest.TestCase):

    @patch('agno_agents.profile_agent.os.getenv')
    @patch('agno_agents.profile_agent.Agent.__init__')
    @patch('agno_agents.profile_agent.get_user_knowledge_base') # Patched where it's looked up for __init__
    def test_initialization_api_key_present(self, mock_get_kb, mock_agent_super_init, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        mock_kb_instance = MagicMock()
        mock_get_kb.return_value = mock_kb_instance

        agent = LearningProfileAgent(user_id="test_user_init_success")

        mock_getenv.assert_called_with("OPENAI_API_KEY")
        mock_agent_super_init.assert_called_once()
        mock_get_kb.assert_called_once_with("test_user_init_success")
        self.assertEqual(agent.user_id, "test_user_init_success")
        self.assertEqual(agent.kb, mock_kb_instance)
        # print("TestLearningProfileAgent.test_initialization_api_key_present: PASSED")


    @patch('agno_agents.profile_agent.os.getenv')
    def test_initialization_api_key_missing(self, mock_getenv):
        mock_getenv.return_value = ""
        with self.assertRaises(ValueError) as context:
            LearningProfileAgent(user_id="test_user_init_fail")
        self.assertIn("OPENAI_API_KEY not found or is a placeholder", str(context.exception))
        # print("TestLearningProfileAgent.test_initialization_api_key_missing: PASSED")


    @patch('agno_agents.profile_agent.os.getenv')
    @patch('agno_agents.profile_agent.get_user_knowledge_base') # Patched for __init__
    def test_analyze_user_profile_no_kb_available(self, mock_get_kb, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        mock_get_kb.return_value = None

        with patch('agno_agents.profile_agent.Agent.__init__'):
            agent = LearningProfileAgent(user_id="test_user_no_kb")

        self.assertIsNone(agent.kb)
        response = agent.analyze_user_profile()
        self.assertIn("Could not analyze user profile: Knowledge Base not available", response)
        # print("TestLearningProfileAgent.test_analyze_user_profile_no_kb_available: PASSED")


    @patch('agno_agents.profile_agent.os.getenv')
    @patch('agno_agents.profile_agent.query_knowledge_base') # Patched where it's looked up for analyze_user_profile
    @patch('agno_agents.profile_agent.get_user_knowledge_base') # Patched for __init__
    def test_analyze_user_profile_kb_empty(self, mock_get_kb_init, mock_query_kb_call, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        mock_kb_instance = MagicMock()
        mock_get_kb_init.return_value = mock_kb_instance
        mock_query_kb_call.return_value = [] # query_knowledge_base call within analyze_user_profile returns empty

        with patch('agno_agents.profile_agent.Agent.__init__'):
             agent = LearningProfileAgent(user_id="test_user_kb_empty")

        response = agent.analyze_user_profile()
        self.assertIn("No documents found", response) # This is the expected message now
        mock_query_kb_call.assert_called_once_with(mock_kb_instance, query_text="", limit=50)
        # print("TestLearningProfileAgent.test_analyze_user_profile_kb_empty: PASSED")


    @patch('agno_agents.profile_agent.os.getenv')
    @patch('agno_agents.profile_agent.query_knowledge_base') # Patched for analyze_user_profile
    @patch('agno_agents.profile_agent.get_user_knowledge_base') # Patched for __init__
    @patch.object(LearningProfileAgent, '__call__')
    def test_analyze_user_profile_success(self, mock_agent_call, mock_get_kb_init, mock_query_kb_call, mock_getenv):
        mock_getenv.return_value = "fake_api_key"

        doc1_content = "User is interested in Python programming."
        mock_doc1 = ActualUtilDocument(content=doc1_content, metadata={'source': 'email'}, id="doc1")
        doc2_content = "Also learning about data science."
        mock_doc2 = ActualUtilDocument(content=doc2_content, metadata={'source': 'web'}, id="doc2")

        mock_kb_instance = MagicMock()
        mock_get_kb_init.return_value = mock_kb_instance
        mock_query_kb_call.return_value = [mock_doc1, mock_doc2]

        expected_llm_response = "Main topics: Python, Data Science."
        mock_agent_call.return_value = expected_llm_response

        with patch('agno_agents.profile_agent.Agent.__init__'):
            agent = LearningProfileAgent(user_id="test_user_success")

        response = agent.analyze_user_profile(max_docs=10, query_str="test query")

        self.assertEqual(response, expected_llm_response)
        mock_query_kb_call.assert_called_once_with(mock_kb_instance, query_text="test query", limit=10)

        expected_concatenated_content = f"{doc1_content}\n\n---\n\n{doc2_content}"
        mock_agent_call.assert_called_once()
        args, _ = mock_agent_call.call_args
        prompt_sent_to_llm = args[0]
        self.assertIn(expected_concatenated_content, prompt_sent_to_llm)
        # print("TestLearningProfileAgent.test_analyze_user_profile_success: PASSED")

    @patch('agno_agents.profile_agent.os.getenv')
    @patch('agno_agents.profile_agent.query_knowledge_base') # Patched for analyze_user_profile
    @patch('agno_agents.profile_agent.get_user_knowledge_base') # Patched for __init__
    @patch.object(LearningProfileAgent, '__call__')
    def test_analyze_user_profile_llm_error(self, mock_agent_call, mock_get_kb_init, mock_query_kb_call, mock_getenv):
        mock_getenv.return_value = "fake_api_key"

        mock_doc1 = ActualUtilDocument(content="Some content.", id="doc_err")
        mock_kb_instance = MagicMock()
        mock_get_kb_init.return_value = mock_kb_instance
        mock_query_kb_call.return_value = [mock_doc1]

        mock_agent_call.side_effect = Exception("LLM API Error")

        with patch('agno_agents.profile_agent.Agent.__init__'):
            agent = LearningProfileAgent(user_id="test_user_llm_error")

        response = agent.analyze_user_profile()
        self.assertIn("Failed to generate profile due to LLM error: LLM API Error", response)
        # print("TestLearningProfileAgent.test_analyze_user_profile_llm_error: PASSED")

    @patch('agno_agents.profile_agent.os.getenv')
    @patch('agno_agents.profile_agent.query_knowledge_base') # Patched for analyze_user_profile
    @patch('agno_agents.profile_agent.get_user_knowledge_base') # Patched for __init__
    def test_analyze_user_profile_kb_docs_no_content(self, mock_get_kb_init, mock_query_kb_call, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        mock_doc_no_content1 = ActualUtilDocument(content="", id="no_content1")
        mock_doc_no_content2 = ActualUtilDocument(content=None, id="no_content2") # Test None content

        mock_kb_instance = MagicMock()
        mock_get_kb_init.return_value = mock_kb_instance
        # query_knowledge_base call within analyze_user_profile returns docs that effectively have no content
        mock_query_kb_call.return_value = [mock_doc_no_content1, mock_doc_no_content2]

        with patch('agno_agents.profile_agent.Agent.__init__'):
             agent = LearningProfileAgent(user_id="test_user_docs_no_content")

        response = agent.analyze_user_profile()
        self.assertIn("Retrieved documents have no text content.", response)
        mock_query_kb_call.assert_called_once_with(mock_kb_instance, query_text="", limit=50)
        # print("TestLearningProfileAgent.test_analyze_user_profile_kb_docs_no_content: PASSED")


if __name__ == '__main__':
    print(f"AGNO_AVAILABLE status (imported by test_profile_agent): {AGNO_AVAILABLE}")
    unittest.main()
