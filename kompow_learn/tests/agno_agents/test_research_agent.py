import unittest
from unittest.mock import patch, MagicMock, ANY
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agno_agents.research_agent import ResearchAgent
# We will mock DuckDuckGoTools and the Agent's __call__ method.

# Since research_agent doesn't directly import from utils.knowledge_base,
# the AGNO_AVAILABLE status is not directly relevant here unless an underlying Agno component fails to import.
# For now, we assume 'agno.agent.Agent' and 'agno.models.openai.OpenAIChat' import fine or are not the issue.


class TestResearchAgent(unittest.TestCase):

    @patch('agno_agents.research_agent.os.getenv')
    def test_initialization_api_key_missing(self, mock_getenv):
        mock_getenv.return_value = "" # Simulate OPENAI_API_KEY is missing
        with self.assertRaises(ValueError) as context:
            ResearchAgent()
        self.assertIn("OPENAI_API_KEY not found or is a placeholder", str(context.exception))
        # print("TestResearchAgent.test_initialization_api_key_missing: PASSED")

    @patch('agno_agents.research_agent.os.getenv')
    @patch('agno_agents.research_agent.DuckDuckGoTools')
    @patch('agno_agents.research_agent.Agent.__init__') # Mock the superclass __init__
    def test_initialization_api_key_present(self, mock_agent_super_init, mock_duckduckgo_tools_constructor, mock_getenv):
        mock_getenv.return_value = "fake_api_key" # Simulate API key is present
        mock_ddg_instance = MagicMock()
        mock_duckduckgo_tools_constructor.return_value = mock_ddg_instance

        agent = ResearchAgent(model_id="test-model", agent_id="test_research_agent")

        mock_getenv.assert_called_with("OPENAI_API_KEY")
        mock_duckduckgo_tools_constructor.assert_called_once_with(num_results=5, id="duckduckgo_search")

        # Check that Agent.__init__ was called with the expected arguments
        # The tools list should contain the instance returned by DuckDuckGoTools constructor
        mock_agent_super_init.assert_called_once()
        args, kwargs = mock_agent_super_init.call_args
        self.assertEqual(kwargs.get('id'), "test_research_agent")
        self.assertIn(mock_ddg_instance, kwargs.get('tools', []))
        self.assertTrue(kwargs.get('show_tool_calls'))
        # print("TestResearchAgent.test_initialization_api_key_present: PASSED")


    @patch('agno_agents.research_agent.os.getenv')
    @patch('agno_agents.research_agent.DuckDuckGoTools')
    @patch('agno_agents.research_agent.Agent.__init__')
    @patch.object(ResearchAgent, '__call__') # Mock the agent's __call__ method (LLM interaction)
    def test_research_topics_success_single_topic_string(self, mock_agent_call, mock_agent_super_init, mock_ddg_tools, mock_getenv):
        mock_getenv.return_value = "fake_api_key"

        expected_research_summary = "Detailed research on Topic X..."
        mock_agent_call.return_value = expected_research_summary

        agent = ResearchAgent() # Superclass init is mocked

        topic_input = "Topic X"
        response = agent.research_topics(topic_input)

        self.assertEqual(response, expected_research_summary)

        # Verify agent.__call__ was invoked with a prompt that includes the topic
        mock_agent_call.assert_called_once()
        args, _ = mock_agent_call.call_args
        prompt_sent_to_llm = args[0]
        self.assertIn(f"'{topic_input}'", prompt_sent_to_llm)
        # print("TestResearchAgent.test_research_topics_success_single_topic_string: PASSED")

    @patch('agno_agents.research_agent.os.getenv')
    @patch('agno_agents.research_agent.DuckDuckGoTools')
    @patch('agno_agents.research_agent.Agent.__init__')
    @patch.object(ResearchAgent, '__call__')
    def test_research_topics_success_list_of_topics(self, mock_agent_call, mock_agent_super_init, mock_ddg_tools, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        expected_summary = "Summary for Topic A and Topic B."
        mock_agent_call.return_value = expected_summary

        agent = ResearchAgent()
        topics_input_list = ["Topic A", "Topic B"]
        topics_input_str = "Topic A, Topic B" # How it will be in the prompt

        response = agent.research_topics(topics_input_list)
        self.assertEqual(response, expected_summary)
        mock_agent_call.assert_called_once()
        args, _ = mock_agent_call.call_args
        prompt_sent_to_llm = args[0]
        self.assertIn(f"'{topics_input_str}'", prompt_sent_to_llm)
        # print("TestResearchAgent.test_research_topics_success_list_of_topics: PASSED")


    @patch('agno_agents.research_agent.os.getenv')
    @patch('agno_agents.research_agent.DuckDuckGoTools')
    @patch('agno_agents.research_agent.Agent.__init__')
    @patch.object(ResearchAgent, '__call__')
    def test_research_topics_llm_error(self, mock_agent_call, mock_agent_super_init, mock_ddg_tools, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        mock_agent_call.side_effect = Exception("LLM API Error")

        agent = ResearchAgent()

        response = agent.research_topics("Some Topic")
        self.assertIn("Failed to conduct research due to an error: LLM API Error", response)
        # print("TestResearchAgent.test_research_topics_llm_error: PASSED")


    @patch('agno_agents.research_agent.os.getenv')
    @patch('agno_agents.research_agent.DuckDuckGoTools')
    @patch('agno_agents.research_agent.Agent.__init__')
    @patch.object(ResearchAgent, '__call__') # Mock the LLM call itself
    def test_research_topics_empty_string_input(self, mock_agent_call, mock_agent_super_init, mock_ddg_tools, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        # Mock what the agent would return if it decided the input was bad *before* an LLM call,
        # OR mock the LLM response if the empty topic string actually goes to the LLM.
        # The current research_topics implementation returns "No topics provided..." before an LLM call.

        agent = ResearchAgent()

        response_empty_topic = agent.research_topics("")
        self.assertEqual(response_empty_topic, "No topics provided for research.")
        mock_agent_call.assert_not_called() # LLM should not be called if input is empty string

        response_space_topic = agent.research_topics("   ")
        self.assertEqual(response_space_topic, "No topics provided for research.")
        mock_agent_call.assert_not_called()
        # print("TestResearchAgent.test_research_topics_empty_input: PASSED")

    @patch('agno_agents.research_agent.os.getenv')
    @patch('agno_agents.research_agent.DuckDuckGoTools')
    @patch('agno_agents.research_agent.Agent.__init__')
    def test_research_topics_invalid_input_type(self, mock_agent_super_init, mock_ddg_tools, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        agent = ResearchAgent()
        response = agent.research_topics(123) # Pass an integer
        self.assertEqual(response, "Invalid input: Topics must be a string or a list of strings.")
        # print("TestResearchAgent.test_research_topics_invalid_input_type: PASSED")


if __name__ == '__main__':
    # This allows running the tests directly from this file
    # Note: AGNO_AVAILABLE from knowledge_base isn't directly used here,
    # but if agno itself had issues, it could affect Agent/OpenAIChat imports.
    # The dummy classes are primarily for knowledge_base.py's direct Agno dependencies.
    unittest.main()
