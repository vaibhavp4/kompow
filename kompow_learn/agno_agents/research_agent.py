import os
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
# Consider adding from agno.message import Message for type hinting if needed

class ResearchAgent(Agent):
    def __init__(self, model_id: str = "gpt-3.5-turbo", agent_id: str = "research_agent", **kwargs):
        self.model_id = model_id

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY not found or is a placeholder. ResearchAgent cannot operate without it.")

        # DuckDuckGoTools typically doesn't require an API key.
        # If using a tool like GoogleSearchTools, you might need to load SERPAPI_API_KEY or similar.
        # e.g., serp_api_key = os.getenv("SERPAPI_API_KEY")
        # if not serp_api_key: print("Warning: SERPAPI_API_KEY not found for Google Search.")

        search_tools = DuckDuckGoTools(num_results=5, id="duckduckgo_search")

        super().__init__(
            id=agent_id,
            role=(
                "You are an AI research assistant. Your primary function is to use web search tools "
                "to gather detailed, up-to-date information on specified topics and keywords. "
                "After conducting research, you must synthesize the findings into a comprehensive summary."
            ),
            instructions=(
                "For each topic or question provided by the user, you must: "
                "1. Utilize your web search tool to find relevant and reliable information. "
                "2. Analyze the search results to extract key facts, explanations, and supporting details. "
                "3. Synthesize this information into a coherent and comprehensive summary for each topic. "
                "4. If possible, mention the source URLs in your summary (though focus on the content). "
                "5. If multiple topics are given, address each one clearly and separately in your final response. "
                "Present the information in a structured and easy-to-understand manner."
            ),
            model=OpenAIChat(id=self.model_id, api_key=api_key),
            tools=[search_tools],
            show_tool_calls=True, # Shows tool interactions in Agno's output
            **kwargs
        )

    def research_topics(self, topics: list[str] | str) -> str:
        """
        Conducts research on a list of topics (or a single topic string) and returns a summary.
        """
        if isinstance(topics, list):
            topics_string = ", ".join(topics)
        elif isinstance(topics, str):
            topics_string = topics
        else:
            return "Invalid input: Topics must be a string or a list of strings."

        if not topics_string.strip():
            return "No topics provided for research."

        # The prompt is crucial. It must guide the LLM to use the tools for the given topics.
        # Agno's agent framework will handle the tool calls based on the LLM's decision.
        prompt_for_llm = (
            f"Please conduct thorough research on the following topic(s): '{topics_string}'.\n"
            f"Use your available search tools to gather information, then provide a detailed summary "
            f"addressing each topic. If multiple topics are listed, ensure each is covered."
        )

        print(f"\nResearchAgent: Initiating research for topics: '{topics_string}' using model {self.model_id}...")

        try:
            # Invoke the agent. Agno's framework will manage tool use based on the LLM's interpretation of the prompt.
            response = self(prompt_for_llm) # __call__ method of Agent

            # The response object might be an Agno Message or just string, depending on Agno version/settings.
            # Assuming it can be cast to string for the summary.
            research_summary = str(response)

            # Check if the LLM indicated it couldn't use tools or find info (a common failure mode)
            if "sorry" in research_summary.lower() and "unable to find information" in research_summary.lower():
                print("Warning: LLM indicated it might have had trouble finding information or using tools.")

        except Exception as e:
            print(f"Error during research process: {e}")
            return f"Failed to conduct research due to an error: {e}"

        return research_summary

if __name__ == "__main__":
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Attempting to load OPENAI_API_KEY from: {dotenv_path}")

    api_key_is_set = os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here"
    if not api_key_is_set:
        print("\nCRITICAL: OPENAI_API_KEY is not set or is a placeholder in your .env file.")
        print("The ResearchAgent requires a valid API key for its LLM to function.")
        print("Please set a valid OPENAI_API_KEY in kompow_learn/.env.")
        exit(1)
    else:
        print("OpenAI API key appears to be set.")

    # No specific API key for DuckDuckGoTools is usually needed.
    # If using other tools like GoogleSearch, add checks for their keys here.

    print(f"\n--- Testing ResearchAgent ---")

    try:
        research_agent = ResearchAgent(model_id="gpt-3.5-turbo") # Or "gpt-4o"
        print(f"ResearchAgent initialized with model {research_agent.model_id} and DuckDuckGoTools.")

        sample_topics = [
            "What are the latest advancements in battery technology for electric vehicles in 2024?",
            "Explain the concept of 'Zero-Knowledge Proofs' in simple terms and list their common applications."
            # "Current status of Starship development by SpaceX" # Another example
        ]
        # Or a single string:
        # sample_topics = "Impact of AI on climate change mitigation strategies."

        print(f"\nTopics for research: {sample_topics}")

        summary_output = research_agent.research_topics(sample_topics)

        print("\n--- Research Summary Result ---")
        # Agno with show_tool_calls=True will print tool interactions before this final summary.
        print(summary_output)

    except ValueError as ve: # Catch API key error from constructor
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during the ResearchAgent test: {e}")

    print("\n--- ResearchAgent Test Complete ---")
