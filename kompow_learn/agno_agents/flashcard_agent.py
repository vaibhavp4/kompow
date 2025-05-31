import os
from dotenv import load_dotenv
import json
import re # Added import for regular expressions

from agno.agent import Agent
from agno.models.openai import OpenAIChat
# from agno.message import Message # For type hinting if needed

class FlashcardGenerationAgent(Agent):
    def __init__(self, model_id: str = "gpt-3.5-turbo", agent_id: str = "flashcard_agent", **kwargs):
        self.model_id = model_id

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY not found or is a placeholder. FlashcardGenerationAgent cannot operate.")

        # Instructions for the agent, including JSON format request
        # Note: Forcing JSON output is more robust with newer models (e.g., gpt-3.5-turbo-1106+, gpt-4-turbo)
        # and by setting response_format={"type": "json_object"} in the model parameters.
        # Agno's OpenAIChat model might need an update to pass this, or it might pass it via extra_body.
        model_params = {"response_format": {"type": "json_object"}} if "1106" in model_id or "turbo" in model_id or "gpt-4" in model_id else {}

        # If Agno's OpenAIChat doesn't directly support response_format,
        # it might be passed via `model_kwargs` or similar, or the prompt needs to be very explicit.
        # For this implementation, we rely on the prompt and Agno's default behavior,
        # but ideally, Agno would allow passing `response_format`.

        super().__init__(
            id=agent_id,
            role=(
                "You are an AI assistant specialized in creating educational flashcards (question and answer pairs) "
                "from a given text. Each flashcard must be concise, focusing on a single, verifiable piece of information "
                "from the text."
            ),
            instructions=(
                "Given the text input, generate a list of flashcards. "
                "Each flashcard must have a 'question' key and an 'answer' key. "
                "The questions should be clear, directly testing understanding of key information from the text. "
                "The answers should be accurate, factual, and derived strictly from the provided text. "
                "Format your entire output as a single JSON object containing a list of these flashcard objects. "
                "For example: {\"flashcards\": [{\"question\": \"What is concept X?\", \"answer\": \"Concept X is defined as...\"}, ...]}"
                "Ensure the output is only the JSON object, with no leading or trailing text."
            ),
            model=OpenAIChat(
                id=self.model_id,
                api_key=api_key
                # model_kwargs cannot be passed to Agno's OpenAIChat as of current understanding.
                # JSON output will rely on prompt engineering.
            ),
            # No tools needed for this agent by default
            **kwargs
        )
        # Print a notice if model_params had content, indicating we can't enforce JSON mode via API.
        if model_params:
             print(f"Notice: For model {self.model_id}, JSON output mode was intended but Agno's OpenAIChat "
                   f"wrapper may not support passing 'response_format'. JSON output relies on prompt adherence.")


    def generate_flashcards_from_text(self, text_content: str, max_flashcards: int = 10) -> list[dict] | str:
        """
        Generates flashcards from the provided text content.
        Returns a list of flashcard dictionaries or the raw LLM response string if JSON parsing fails.
        """
        if not text_content or not text_content.strip():
            return "Input text is empty. Cannot generate flashcards."

        # The detailed JSON instructions are in the agent's system prompt (instructions).
        # Here, we just provide the text.
        prompt_for_llm = (
            f"Please generate up to {max_flashcards} flashcards in the specified JSON format "
            f"(a JSON object with a single key \"flashcards\" containing a list of question/answer objects) "
            f"based on the following text content:\n\n"
            f"--- TEXT START ---\n"
            f"{text_content}\n"
            f"--- TEXT END ---\n\n"
            f"Remember to only output the JSON object."
        )

        print(f"\nFlashcardAgent: Generating flashcards from text (length: {len(text_content)} chars) using model {self.model_id}...")

        try:
            response = self(prompt_for_llm) # Invoke the agent
            response_text = str(response).strip() # Get the raw text output

            # Attempt to find JSON block if the LLM includes markdown or other text
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            else: # If no markdown, assume the whole response is the JSON or needs cleaning
                # Sometimes LLMs might add "Here is the JSON output:"
                if response_text.startswith('{') and response_text.endswith('}'):
                    pass # Looks like JSON
                else: # Attempt to extract the first '{' to the last '}'
                    first_brace = response_text.find('{')
                    last_brace = response_text.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        response_text = response_text[first_brace : last_brace+1]
                    else:
                        print("Warning: LLM response does not appear to be a JSON object string.")


            parsed_json = json.loads(response_text)

            if isinstance(parsed_json, dict) and "flashcards" in parsed_json and isinstance(parsed_json["flashcards"], list):
                flashcards = parsed_json["flashcards"]
                # Validate basic structure of each flashcard
                for card in flashcards:
                    if not (isinstance(card, dict) and "question" in card and "answer" in card):
                        print(f"Warning: Invalid flashcard structure found: {card}")
                        # Fallback or filter out invalid cards if necessary
                return flashcards
            else:
                print("Warning: JSON output is not in the expected format (e.g., {\"flashcards\": [...]}).")
                return f"JSON parsing error: Unexpected structure. Raw output: {response_text}"

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response: {e}")
            print(f"Raw LLM response was:\n{response_text}")
            return f"Failed to parse JSON. Raw LLM output: {response_text}"
        except Exception as e:
            print(f"An unexpected error occurred during flashcard generation: {e}")
            return f"Failed to generate flashcards due to an error: {e}"

if __name__ == "__main__":
    # Need to go up two levels from agno_agents to reach project root for .env
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Attempting to load OPENAI_API_KEY from: {dotenv_path}")
    # For regex during parsing:
    import re


    api_key_is_set = os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here"
    if not api_key_is_set:
        print("\nCRITICAL: OPENAI_API_KEY is not set or is a placeholder in your .env file.")
        print("The FlashcardGenerationAgent requires a valid API key for its LLM to function.")
        print("Please set a valid OPENAI_API_KEY in kompow_learn/.env.")
        exit(1)
    else:
        print("OpenAI API key appears to be set.")

    print(f"\n--- Testing FlashcardGenerationAgent ---")

    try:
        # Using a model known to be better at following JSON instructions if possible
        flashcard_agent = FlashcardGenerationAgent(model_id="gpt-3.5-turbo-1106")
        print(f"FlashcardGenerationAgent initialized with model {flashcard_agent.model_id}.")

        sample_text_content = (
            "The mitochondria is often called the powerhouse of the cell. It generates most of the cell's supply of adenosine triphosphate (ATP), "
            "used as a source of chemical energy. The process of ATP generation in mitochondria is known as cellular respiration. "
            "Photosynthesis, on the other hand, occurs in chloroplasts in plant cells, converting light energy into chemical energy."
            "The Earth revolves around the Sun, taking approximately 365.25 days to complete one orbit, which is why we have leap years."
        )

        print(f"\nSample text for flashcard generation:\n{sample_text_content}\n")

        flashcards_output = flashcard_agent.generate_flashcards_from_text(sample_text_content, max_flashcards=5)

        print("\n--- Flashcard Generation Result ---")
        if isinstance(flashcards_output, list):
            if not flashcards_output:
                print("No flashcards were generated.")
            else:
                print(f"Successfully generated {len(flashcards_output)} flashcards:")
                for i, card in enumerate(flashcards_output):
                    print(f"\nFlashcard {i+1}:")
                    print(f"  Q: {card.get('question')}")
                    print(f"  A: {card.get('answer')}")
        else: # It's a string, meaning an error or raw output
            print("Failed to generate structured flashcards. Output:")
            print(flashcards_output)

    except ValueError as ve: # Catch API key error from constructor
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during the FlashcardGenerationAgent test: {e}")

    print("\n--- FlashcardGenerationAgent Test Complete ---")
