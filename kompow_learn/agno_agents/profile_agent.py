import os
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
# from agno.tools.misc import CodeInterpreter # Example if tools were needed

from utils.knowledge_base import get_user_knowledge_base, query_knowledge_base, ActualAgnoDocument as Document # Use the (potentially dummy) Document

# Define a placeholder for get_all_documents if needed.
# For now, we'll rely on query_knowledge_base with a broad query.
# def get_all_documents_from_kb(kb) -> list[Document]:
#     # This would be a new function in knowledge_base.py if query_knowledge_base isn't sufficient
#     # For LanceDB, it might involve:
#     # table = kb.vector_db._load_table() # Accessing underlying table
#     # all_data = table.to_arrow().to_pydict() # Or similar to get all records
#     # return [Document(id=row['id'], content=row['text'], metadata=row['metadata']) for row in all_data] # Reconstruct
#     print("Placeholder: get_all_documents_from_kb would fetch all documents.")
#     return []

class LearningProfileAgent(Agent):
    def __init__(self, user_id: str, model_id: str = "gpt-3.5-turbo", agent_id: str = None, **kwargs):
        self.user_id = user_id
        self.model_id = model_id # Allow model to be specified

        # Load API key for the model
        # Note: Agno's OpenAIChat might load this automatically if OPENAI_API_KEY is set,
        # but explicit loading ensures clarity and allows for checks.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            # This agent relies heavily on the LLM, so if no key, it can't do much.
            raise ValueError("OPENAI_API_KEY not found or is a placeholder. LearningProfileAgent cannot operate.")

        super().__init__(
            id=agent_id or f"learning_profile_agent_{user_id}",
            role=(
                "You are an AI assistant specializing in analyzing a user's stored text documents "
                "to identify their learning interests, key topics, and areas of expertise. "
                "Focus on extracting actionable insights about what the user is trying to learn or knows well."
            ),
            instructions=(
                "Based on the provided documents, synthesize a profile of the user's learning interests. "
                "Identify the main topics and concepts the user seems to be focused on. "
                "List specific keywords or technologies mentioned frequently. "
                "If possible, infer potential learning goals or areas of deep knowledge. "
                "Present the output as a structured summary."
            ),
            model=OpenAIChat(id=self.model_id, api_key=api_key), # Pass API key explicitly if required by Agno version
            # tools=[CodeInterpreter(id="code_interpreter")] # Example, not used yet
            knowledge_enabled=False, # This agent uses its own KB access logic, not Agno's built-in agent KB
            **kwargs
        )
        self.kb = get_user_knowledge_base(self.user_id)
        if not self.kb:
            print(f"Warning: KnowledgeBase could not be initialized for user {self.user_id}. Analysis will be limited.")
            # Depending on strictness, could raise an error here too.

    def analyze_user_profile(self, max_docs: int = 50, query_str: str = "") -> str:
        """
        Analyzes the user's documents from their Knowledge Base to generate a learning profile.
        A broad query string (e.g., empty or a very general term) can be used to fetch diverse documents.
        """
        if not self.kb:
            return "Could not analyze user profile: Knowledge Base not available."

        print(f"Attempting to retrieve documents for user {self.user_id} with query: '{query_str if query_str else 'broad (all docs)'}'")

        # Fetch documents from the user's KB.
        # Using a generic query and a high limit to simulate fetching "all" or "most relevant" documents.
        # A more sophisticated approach might be needed for very large KBs.
        documents: list[Document] = query_knowledge_base(self.kb, query_text=query_str, limit=max_docs)

        if not documents:
            return "No documents found in the user's knowledge base. Cannot analyze profile."

        # Concatenate content from all documents.
        # Consider token limits for the LLM. A more advanced version might summarize chunks.
        full_text_content = "\n\n---\n\n".join([doc.content for doc in documents if doc.content])

        if not full_text_content.strip():
            return "Retrieved documents have no text content. Cannot analyze profile."

        # Rough token estimation (very approximate)
        # avg_chars_per_token = 4
        # num_tokens = len(full_text_content) / avg_chars_per_token
        # print(f"Estimated number of tokens in content: {num_tokens}")
        # max_tokens_for_model = 8000 # Example for gpt-3.5-turbo (16k context, but leave room for prompt & response)
        # if num_tokens > max_tokens_for_model:
        #     print(f"Warning: Content is very long ({num_tokens} tokens), may exceed model limits. Consider summarizing or chunking.")
            # For now, we'll send it all, but this is where summarization/map-reduce could be added.

        prompt_for_llm = (
            f"The following text is a collection of documents from a user's knowledge base. "
            f"Please analyze this content and generate a learning profile summary. Focus on:\n"
            f"1. Main topics and concepts of interest.\n"
            f"2. Specific keywords, technologies, or skills mentioned.\n"
            f"3. Potential learning goals or areas of deep knowledge.\n"
            f"4. Any recurring themes or questions.\n\n"
            f"Present this as a structured summary.\n\n"
            f"User's Documents:\n{'-'*20}\n{full_text_content}\n{'-'*20}\n"
            f"End of User's Documents. Begin Profile Summary:\n"
        )

        print(f"\nSending content from {len(documents)} documents to LLM for analysis for user {self.user_id}...")

        # Use the agent's built-in response generation
        # The `self.run(prompt_for_llm)` or `self(prompt_for_llm)` is typical for Agno agents.
        # `self.print_response` is often a utility to run and print. Let's assume `self.run` is the core.
        try:
            # response = self.run(prompt_for_llm) # If .run() is the method to get LLM response
            # For simple chat, often it's just calling the agent instance
            response = self(prompt_for_llm)
            # If the response is an Agno Message object, extract its content:
            # analysis_result = response.content if hasattr(response, 'content') else str(response)
            analysis_result = str(response) # Assuming __str__ of response gives the text
        except Exception as e:
            print(f"Error during LLM interaction: {e}")
            return f"Failed to generate profile due to LLM error: {e}"

        return analysis_result

if __name__ == "__main__":
    # Load .env file from the root of the project (kompow_learn directory)
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Attempting to load OPENAI_API_KEY from: {dotenv_path}")

    api_key_is_set = os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here"
    if not api_key_is_set:
        print("\nCRITICAL: OPENAI_API_KEY is not set or is a placeholder in your .env file.")
        print("The LearningProfileAgent requires a valid API key to function.")
        print("Please set a valid OPENAI_API_KEY in kompow_learn/.env.")
        # Exit if no key, as the agent is useless without it.
        exit(1)
    else:
        print("OpenAI API key appears to be set.")

    # Use a test user_id that should have data from previous email parsing tests.
    # This needs to match a user_id for whom data was stored in test_email_parser.py
    # Typically, this would be the sender of one of the test emails.
    # For example, if your .env has EMAIL_USER=kompow_learner@example.com, and this user sent emails
    # or if you used a default_user_id like "shared_kompow_user" in email_parser.py

    # !!! IMPORTANT: Replace this with a user_id that actually has data in your LanceDB store !!!
    # You might need to run email_parser.py first with your .env configured to populate data.
    # For example, if your EMAIL_USER in .env is 'test@example.com' and it sent emails
    # that were processed and stored.
    test_user_with_data = os.getenv("EMAIL_USER", "test_user@example.com") # Fallback for safety
    # If you used a fixed default user in email_parser's main block when no sender was found:
    # test_user_with_data = "shared_kompow_user"

    print(f"\n--- Testing LearningProfileAgent for user: {test_user_with_data} ---")

    try:
        profile_agent = LearningProfileAgent(user_id=test_user_with_data, model_id="gpt-3.5-turbo") # or "gpt-4o"
        print(f"LearningProfileAgent for {test_user_with_data} initialized with model {profile_agent.model_id}.")

        # Analyze profile using a broad query (empty string) to get diverse documents
        # You can also try specific queries like "python programming" if you know such content exists
        # The query here is for the *vector search* in KB, not the LLM prompt.
        analysis_output = profile_agent.analyze_user_profile(max_docs=20, query_str="")

        print("\n--- Learning Profile Analysis Result ---")
        print(analysis_output)

    except ValueError as ve: # Catch API key error from constructor
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during the test: {e}")

    print("\n--- LearningProfileAgent Test Complete ---")
