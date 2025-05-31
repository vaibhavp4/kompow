import unittest
from unittest.mock import patch, MagicMock, ANY
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agno_agents.flashcard_agent import FlashcardGenerationAgent
from agno.models.openai import OpenAIChat as AgnoOpenAIChat

class TestFlashcardGenerationAgent(unittest.TestCase):

    @patch('agno_agents.flashcard_agent.os.getenv')
    def test_initialization_api_key_missing(self, mock_getenv):
        mock_getenv.return_value = ""
        with self.assertRaises(ValueError) as context:
            FlashcardGenerationAgent()
        self.assertIn("OPENAI_API_KEY not found or is a placeholder", str(context.exception))

    @patch('agno_agents.flashcard_agent.os.getenv')
    @patch('agno_agents.flashcard_agent.OpenAIChat')
    @patch('agno_agents.flashcard_agent.Agent.__init__')
    def test_initialization_api_key_present_default_model(self, mock_agent_super_init, mock_openai_chat_constructor, mock_getenv):
        mock_getenv.return_value = "fake_api_key"

        # Test the agent's internal logic for setting model_params
        agent = FlashcardGenerationAgent() # Default model_id="gpt-3.5-turbo"

        mock_openai_chat_constructor.assert_called_once()
        args, kwargs = mock_openai_chat_constructor.call_args
        self.assertEqual(kwargs.get('id'), "gpt-3.5-turbo")
        self.assertEqual(kwargs.get('api_key'), "fake_api_key")
        # Assert that model_kwargs is NOT passed if OpenAIChat doesn't support it
        self.assertNotIn('model_kwargs', kwargs)

        # Check the agent's own model_params variable (if needed to verify logic)
        # This requires knowing that model_params is stored on the instance or checking the logic
        # For "gpt-3.5-turbo", the agent's internal model_params should include response_format
        # This part of the test might be too white-box if model_params isn't an attribute.
        # The print statement in __init__ about model_params will indicate its status.
        # For now, let's assume the important part is that OpenAIChat is called correctly without model_kwargs.

        mock_agent_super_init.assert_called_once()


    @patch('agno_agents.flashcard_agent.os.getenv')
    @patch('agno_agents.flashcard_agent.OpenAIChat')
    @patch('agno_agents.flashcard_agent.Agent.__init__')
    def test_initialization_with_json_supporting_model(self, mock_agent_super_init, mock_openai_chat_constructor, mock_getenv):
        mock_getenv.return_value = "fake_api_key"

        agent = FlashcardGenerationAgent(model_id="gpt-3.5-turbo-1106")

        mock_openai_chat_constructor.assert_called_once()
        args, kwargs = mock_openai_chat_constructor.call_args
        self.assertEqual(kwargs.get('id'), "gpt-3.5-turbo-1106")
        self.assertEqual(kwargs.get('api_key'), "fake_api_key")
        # Assert that model_kwargs is NOT passed
        self.assertNotIn('model_kwargs', kwargs)

        # The agent's internal model_params variable should have been set correctly.
        # The print statement in agent's __init__ would confirm this:
        # "Notice: For model gpt-3.5-turbo-1106, JSON output mode was intended but Agno's OpenAIChat "
        # "wrapper may not support passing 'response_format'. JSON output relies on prompt adherence."

        mock_agent_super_init.assert_called_once()

    @patch('agno_agents.flashcard_agent.os.getenv')
    @patch('agno_agents.flashcard_agent.Agent.__init__')
    @patch.object(FlashcardGenerationAgent, '__call__')
    def test_generate_flashcards_success_valid_json(self, mock_agent_call, mock_super_init, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        agent = FlashcardGenerationAgent()

        researched_text = "Mitochondria is the powerhouse of the cell."
        expected_flashcards_list = [{"question": "What is Mitochondria?", "answer": "The powerhouse of the cell."}]
        llm_response_json_str = json.dumps({"flashcards": expected_flashcards_list})
        mock_agent_call.return_value = llm_response_json_str

        result = agent.generate_flashcards_from_text(researched_text)

        self.assertEqual(result, expected_flashcards_list)
        mock_agent_call.assert_called_once()
        args, _ = mock_agent_call.call_args
        prompt_sent_to_llm = args[0]
        self.assertIn(researched_text, prompt_sent_to_llm)

    @patch('agno_agents.flashcard_agent.os.getenv')
    @patch('agno_agents.flashcard_agent.Agent.__init__')
    @patch.object(FlashcardGenerationAgent, '__call__')
    def test_generate_flashcards_success_json_with_markdown_fences(self, mock_agent_call, mock_super_init, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        agent = FlashcardGenerationAgent()
        expected_flashcards_list = [{"question": "Q1", "answer": "A1"}]
        llm_response_json_str = f"```json\n{json.dumps({'flashcards': expected_flashcards_list})}\n```"
        mock_agent_call.return_value = llm_response_json_str

        result = agent.generate_flashcards_from_text("Some text")
        self.assertEqual(result, expected_flashcards_list)

    @patch('agno_agents.flashcard_agent.os.getenv')
    @patch('agno_agents.flashcard_agent.Agent.__init__')
    @patch.object(FlashcardGenerationAgent, '__call__')
    def test_generate_flashcards_llm_returns_malformed_json(self, mock_agent_call, mock_super_init, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        agent = FlashcardGenerationAgent()
        malformed_json_str = '{"flashcards": [{"question": "Q1", "answer": "A1"}'
        mock_agent_call.return_value = malformed_json_str

        result = agent.generate_flashcards_from_text("text")
        self.assertIsInstance(result, str)
        self.assertIn("Failed to parse JSON", result)
        self.assertIn(malformed_json_str, result)

    @patch('agno_agents.flashcard_agent.os.getenv')
    @patch('agno_agents.flashcard_agent.Agent.__init__')
    @patch.object(FlashcardGenerationAgent, '__call__')
    def test_generate_flashcards_llm_returns_json_wrong_structure(self, mock_agent_call, mock_super_init, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        agent = FlashcardGenerationAgent()
        wrong_structure_json_str = json.dumps({"data": [{"q": "Q1", "a": "A1"}]})
        mock_agent_call.return_value = wrong_structure_json_str

        result = agent.generate_flashcards_from_text("text")
        self.assertIsInstance(result, str)
        self.assertIn("JSON parsing error: Unexpected structure", result)

    @patch('agno_agents.flashcard_agent.os.getenv')
    @patch('agno_agents.flashcard_agent.Agent.__init__')
    @patch.object(FlashcardGenerationAgent, '__call__')
    def test_generate_flashcards_llm_returns_non_json_text(self, mock_agent_call, mock_super_init, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        agent = FlashcardGenerationAgent()
        non_json_response = "Sorry, I cannot generate flashcards at this moment."
        mock_agent_call.return_value = non_json_response

        result = agent.generate_flashcards_from_text("text")
        self.assertIsInstance(result, str)
        self.assertIn("Failed to parse JSON", result)
        self.assertIn(non_json_response, result)


    @patch('agno_agents.flashcard_agent.os.getenv')
    @patch('agno_agents.flashcard_agent.Agent.__init__')
    @patch.object(FlashcardGenerationAgent, '__call__')
    def test_generate_flashcards_llm_call_raises_exception(self, mock_agent_call, mock_super_init, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        agent = FlashcardGenerationAgent()
        mock_agent_call.side_effect = Exception("LLM API Error")

        result = agent.generate_flashcards_from_text("text")
        self.assertIsInstance(result, str)
        self.assertIn("Failed to generate flashcards due to an error: LLM API Error", result)

    @patch('agno_agents.flashcard_agent.os.getenv')
    @patch('agno_agents.flashcard_agent.Agent.__init__')
    @patch.object(FlashcardGenerationAgent, '__call__')
    def test_generate_flashcards_empty_input_text(self, mock_agent_call, mock_super_init, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        agent = FlashcardGenerationAgent()

        result_empty = agent.generate_flashcards_from_text("")
        self.assertEqual(result_empty, "Input text is empty. Cannot generate flashcards.")
        mock_agent_call.assert_not_called()

        result_space = agent.generate_flashcards_from_text("   ")
        self.assertEqual(result_space, "Input text is empty. Cannot generate flashcards.")
        mock_agent_call.assert_not_called()

if __name__ == '__main__':
    unittest.main()
