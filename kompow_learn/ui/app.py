import gradio as gr
import os
import sys
from dotenv import load_dotenv
import json
import traceback

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agno_agents.research_agent import ResearchAgent
from agno_agents.flashcard_agent import FlashcardGenerationAgent
from utils.knowledge_base import (
    get_user_knowledge_base,
    add_flashcard_set_to_kb,
    get_flashcard_sets_for_user,
    get_available_flashcard_topics
)

# --- Global Variables & Agent Initialization ---
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY_ERROR_MESSAGE = (
    "CRITICAL ERROR: OPENAI_API_KEY is not set or is a placeholder in your .env file. "
    "The application cannot function without a valid OpenAI API key for agent operations and semantic search in Knowledge Base. "
    "Please set it in the .env file at the root of the project and restart the application."
)

research_agent_instance: ResearchAgent | None = None
flashcard_agent_instance: FlashcardGenerationAgent | None = None
ON_DEMAND_USER_ID = "on_demand_flashcard_user" # User ID for UI-generated flashcards
kb_for_on_demand_flashcards: 'KnowledgeBase | None' = None
initialization_error: str | None = None

try:
    # Initialize KB first, as agent initialization might depend on a working KB if they use it directly
    # However, our agents currently get KB passed or create it internally.
    # For UI-specific KB ops (like storing flashcards from UI), initialize it here.
    kb_for_on_demand_flashcards = get_user_knowledge_base(ON_DEMAND_USER_ID)
    if kb_for_on_demand_flashcards:
        print(f"KnowledgeBase for '{ON_DEMAND_USER_ID}' initialized for UI operations (storing/viewing flashcards).")
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
             # This specific KB might still work for adding flashcards if schema allows non-embedded content,
             # but listing topics or flashcards via semantic search (current KB impl) will fail.
             warning_kb_no_api = f"Warning: OPENAI_API_KEY is missing. KB for '{ON_DEMAND_USER_ID}' initialized, but operations requiring embeddings (like listing topics/flashcards as currently implemented) will fail."
             print(warning_kb_no_api)
             initialization_error = (initialization_error + "\n" + warning_kb_no_api) if initialization_error else warning_kb_no_api
    else:
        kb_init_fail_msg = f"Critical Warning: KnowledgeBase for '{ON_DEMAND_USER_ID}' could NOT be initialized. Storing and viewing flashcards will fail."
        print(kb_init_fail_msg)
        initialization_error = (initialization_error + "\n" + kb_init_fail_msg) if initialization_error else kb_init_fail_msg


    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        # This is the primary error if API key is missing, as agents need it.
        agents_init_fail_msg = API_KEY_ERROR_MESSAGE
        print(agents_init_fail_msg)
        initialization_error = (initialization_error + "\n" + agents_init_fail_msg) if initialization_error else agents_init_fail_msg
    else:
        research_agent_instance = ResearchAgent(model_id="gpt-3.5-turbo")
        flashcard_agent_instance = FlashcardGenerationAgent(model_id="gpt-3.5-turbo-1106")
        print("Research and Flashcard agents initialized successfully with API key.")

except ValueError as ve:
    error_msg_val = f"Agent Initialization Error: {ve}. Check your OPENAI_API_KEY."
    print(error_msg_val)
    initialization_error = (initialization_error + "\n" + error_msg_val) if initialization_error else error_msg_val
except Exception as e:
    error_msg_exc = f"An unexpected error occurred during agent/KB initialization: {e}"
    print(error_msg_exc)
    traceback.print_exc()
    initialization_error = (initialization_error + "\n" + error_msg_exc) if initialization_error else error_msg_exc


def format_flashcards_html(flashcards: list[dict], title_prefix: str = "Generated") -> str:
    if not flashcards:
        return f"<p>No flashcards to display for {title_prefix.lower()}.</p>"

    html_parts = ["<div style='font-family: sans-serif;'>"]
    html_parts.append(f"<h2>{title_prefix} Flashcards:</h2><hr>")
    for i, card in enumerate(flashcards):
        question = card.get('question', 'N/A')
        answer = card.get('answer', 'N/A')
        html_parts.append(
            f"<div style='margin-bottom: 20px; padding: 10px; border: 1px solid #eee; border-radius: 5px;'>"
            f"<p><b>Flashcard {i+1}</b></p>"
            f"<p><b>Question:</b> {gr.Textbox.postprocess_example(question)}</p>"
            f"<p><b>Answer:</b> {gr.Textbox.postprocess_example(answer)}</p>"
            f"</div>"
        )
    html_parts.append("</div>")
    return "".join(html_parts)

def format_status_html(message: str, details: str = "", msg_type: str = "info") -> str:
    color = "red" if msg_type == "error" else ("darkorange" if msg_type == "warning" else "green" if msg_type == "success" else "dodgerblue")
    border_color = color
    title = msg_type.capitalize()

    details_html = f"<p><small>Details: {gr.Textbox.postprocess_example(details)}</small></p>" if details else ""
    return (
        f"<div style='color: {color}; padding: 10px; border: 1px solid {border_color}; border-radius: 5px; margin-top:10px;'>"
        f"<p><b>{title}:</b> {gr.Textbox.postprocess_example(message)}</p>"
        f"{details_html}"
        f"</div>"
    )

def generate_flashcards_for_topic_ui(topic_text: str):
    print(f"\nUI Action: Generate Flashcards for topic: '{topic_text}'")
    status_messages = []
    final_flashcards_html = None

    if initialization_error and "OPENAI_API_KEY" in initialization_error: # Prioritize API key error for agents
        return None, format_status_html("Application Initialization Failed", initialization_error, "error")
    if not research_agent_instance or not flashcard_agent_instance:
         return None, format_status_html("Agents not available", "AI agents failed to initialize. Check API key or server logs.", "error")
    if not topic_text or not topic_text.strip():
        return None, format_status_html("Input Error", "Topic text cannot be empty.", "error")

    try:
        status_messages.append(format_status_html(f"Starting research for topic: '{topic_text}'...", msg_type="info"))
        researched_text = research_agent_instance.research_topics(topic_text)
        if not researched_text or "Failed to conduct research" in researched_text or len(researched_text) < 50:
            fail_msg = researched_text or "No meaningful content from research."
            status_messages.append(format_status_html("Research Failed", fail_msg, "error"))
            return None, "".join(status_messages)

        status_messages.append(format_status_html(f"Research complete. Content length: {len(researched_text)} chars.", msg_type="info"))
        status_messages.append(format_status_html("Generating flashcards from researched content...", msg_type="info"))
        flashcards_output = flashcard_agent_instance.generate_flashcards_from_text(researched_text)

        if isinstance(flashcards_output, list):
            if not flashcards_output:
                status_messages.append(format_status_html("No flashcards generated.", "Agent returned an empty list.", "warning"))
                final_flashcards_html = format_flashcards_html([], title_prefix="Newly Generated (None)")
            else:
                final_flashcards_html = format_flashcards_html(flashcards_output, title_prefix="Newly Generated")
                status_messages.append(format_status_html(f"Successfully generated {len(flashcards_output)} flashcards!", msg_type="success"))

                if kb_for_on_demand_flashcards:
                    flashcards_json_str = json.dumps(flashcards_output)
                    store_success = add_flashcard_set_to_kb(
                        kb_for_on_demand_flashcards, ON_DEMAND_USER_ID, topic_text,
                        flashcards_json_str, source="on_demand_ui_generation"
                    )
                    if store_success:
                        status_messages.append(format_status_html("Flashcards stored in Knowledge Base.", msg_type="info"))
                    else:
                        status_messages.append(format_status_html("Failed to store flashcards in KB.", "This might be due to KB init issues or API key problems if embeddings are attempted by the KB.", "warning"))
                else:
                    status_messages.append(format_status_html("Flashcard storage skipped.", "KB for on-demand user not initialized.", "warning"))
            return final_flashcards_html, "".join(status_messages)
        else:
            status_messages.append(format_status_html("Flashcard Generation Failed", flashcards_output, "error"))
            return None, "".join(status_messages)
    except Exception as e:
        traceback.print_exc()
        status_messages.append(format_status_html("Critical Application Error", str(e), "error"))
        return None, "".join(status_messages)

def ui_populate_topic_dropdown():
    print("\nUI Action: Populate Topic Dropdown")
    if not kb_for_on_demand_flashcards:
        return gr.Dropdown.update(choices=[], value=None), format_status_html("Cannot load topics", "Knowledge Base for on-demand user not available.", "error")
    if initialization_error and "OPENAI_API_KEY" in initialization_error and not kb_for_on_demand_flashcards.vector_db.embedder:
        # If KB init was fine but embedder (needed for current get_available_flashcard_topics) is missing
        return gr.Dropdown.update(choices=[], value=None), format_status_html("Cannot load topics", "Knowledge Base requires OpenAI API key for current topic retrieval method.", "warning")

    try:
        topics = get_available_flashcard_topics(kb_for_on_demand_flashcards, ON_DEMAND_USER_ID)
        if not topics:
             return gr.Dropdown.update(choices=[], value=None), format_status_html("No stored topics found.", "Generate some flashcards first, or check KB initialization.", "info")
        return gr.Dropdown.update(choices=topics, value=topics[0] if topics else None), format_status_html(f"Found {len(topics)} stored topic(s).", msg_type="success")
    except Exception as e:
        traceback.print_exc()
        return gr.Dropdown.update(choices=[], value=None), format_status_html("Error loading topics", str(e), "error")


def ui_display_stored_flashcards(selected_topic: str):
    print(f"\nUI Action: Display Stored Flashcards for topic: '{selected_topic}'")
    if not selected_topic:
        return format_flashcards_html([], title_prefix=f"Stored (No topic selected)"), format_status_html("Please select a topic.", msg_type="info")
    if not kb_for_on_demand_flashcards:
        return None, format_status_html("Cannot load flashcards", "Knowledge Base for on-demand user not available.", "error")
    if initialization_error and "OPENAI_API_KEY" in initialization_error and not kb_for_on_demand_flashcards.vector_db.embedder:
         return None, format_status_html("Cannot load flashcards", "Knowledge Base requires OpenAI API key for current flashcard retrieval method.", "warning")

    try:
        flashcard_docs = get_flashcard_sets_for_user(kb_for_on_demand_flashcards, ON_DEMAND_USER_ID, topic=selected_topic, limit=5)
        if not flashcard_docs:
            return format_flashcards_html([], title_prefix=f"Stored ({selected_topic})"), format_status_html(f"No stored flashcard sets found for topic: '{selected_topic}'.", msg_type="info")

        all_q_a_pairs = []
        for doc_idx, doc in enumerate(flashcard_docs):
            try:
                flashcards_in_set = json.loads(doc.content)
                # Could add a sub-header here like: f"<h4>Set from {doc.metadata.get('creation_date')} (Source: {doc.metadata.get('source')})</h4>"
                all_q_a_pairs.extend(flashcards_in_set)
            except json.JSONDecodeError:
                print(f"Error decoding JSON for stored flashcard doc ID {doc.id}")
                # Optionally add a message about this specific set failing to parse

        if not all_q_a_pairs: # If sets were found but all failed to parse (unlikely if add_flashcard_set_to_kb validates)
             return format_flashcards_html([], title_prefix=f"Stored ({selected_topic})"), format_status_html(f"Found sets for '{selected_topic}', but could not parse their content.", "warning")

        return format_flashcards_html(all_q_a_pairs, title_prefix=f"Stored ({selected_topic})"), format_status_html(f"Displayed flashcards for '{selected_topic}'. Found {len(flashcard_docs)} set(s), showing content from them.", msg_type="success")
    except Exception as e:
        traceback.print_exc()
        return None, format_status_html(f"Error displaying stored flashcards for '{selected_topic}'", str(e), "error")


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky)) as iface:
    gr.Markdown(
        "<div align='center'><h1>⚡ KompowLearn - Your AI Learning Assistant ⚡</h1></div>"
        "<p align='center'>Generate new flashcards on any topic using AI research, or view your previously generated & stored flashcards.<br>"
        "<i><b>Important:</b> Full functionality (agent operations, semantic search for stored flashcards) requires a valid <code>OPENAI_API_KEY</code> in the <code>.env</code> file.</i></p>"
    )

    # Display general initialization error if any occurred (especially API key for agents)
    # This will be visible regardless of tab.
    if initialization_error and ("OPENAI_API_KEY" in initialization_error or "Agent Initialization Error" in initialization_error):
         gr.HTML(format_status_html("Critical Application Setup Issue", initialization_error, "error"))

    with gr.Tabs() as tabs:
        with gr.TabItem("🧠 Generate New Flashcards", id="generate_tab"):
            gr.Markdown("## Step 1: Enter a Topic for AI Research & Flashcard Creation")
            with gr.Row():
                topic_input_generate = gr.Textbox(
                    lines=3, label="Topic:", placeholder="e.g., 'The basics of Quantum Computing and its common applications'",
                    elem_id="topic_input_generate_textbox"
                )
            submit_button_generate = gr.Button("🚀 Generate & Store Flashcards!", variant="primary")
            gr.Markdown("---")
            with gr.Accordion("✨ Newly Generated Flashcards ✨", open=True):
                flashcards_display_generate = gr.HTML(label="Flashcards Output")
            with gr.Accordion("📊 Generation Status & Logs 📊", open=True):
                status_display_generate = gr.HTML(label="Process Log & Status")

            gr.Examples(
                examples=[
                    "Key concepts of General Relativity", "How do vaccines work to protect the body?",
                    "The history of the Silk Road and its impact on trade",
                    "Explain the main sustainable energy sources and their pros/cons",
                    "What is CRISPR gene editing technology?"],
                inputs=[topic_input_generate], elem_id="generate_examples"
            )

        with gr.TabItem("📚 View Stored Flashcards", id="view_tab"):
            gr.Markdown("## Step 1: Load Available Topics from your Knowledge Base")
            # Note: kb_for_on_demand_flashcards is used here for the ON_DEMAND_USER_ID
            # This section's functionality for loading topics/flashcards might be limited if API key is missing,
            # because get_available_flashcard_topics and get_flashcard_sets_for_user currently use semantic search.
            if not kb_for_on_demand_flashcards:
                 gr.HTML(format_status_html("Knowledge Base Error", f"KB for '{ON_DEMAND_USER_ID}' not initialized. Cannot view stored flashcards.", "error"))
            elif not kb_for_on_demand_flashcards.vector_db.embedder: # Explicitly check for embedder
                 gr.HTML(format_status_html("API Key Advisory", "OpenAI API Key is likely missing or invalid. Listing topics and viewing stored flashcards (which currently uses semantic search) may not work.", "warning"))


            topic_dropdown_stored = gr.Dropdown(label="Select Topic to View Flashcards", interactive=True)
            load_topics_button = gr.Button("🔄 Refresh/Load Topics")

            gr.Markdown("## Step 2: View Flashcards for Selected Topic")
            load_flashcards_button = gr.Button("🔍 Load Flashcards for this Topic")

            with gr.Accordion("📖 Stored Flashcards 📖", open=True):
                flashcards_display_stored = gr.HTML(label="Stored Flashcards")
            with gr.Accordion("📈 Retrieval Status & Logs 📈", open=True):
                status_display_stored = gr.HTML(label="Retrieval Log & Status")

    # Wire up components
    submit_button_generate.click(
        fn=generate_flashcards_for_topic_ui,
        inputs=[topic_input_generate],
        outputs=[flashcards_display_generate, status_display_generate],
        api_name="generate_and_store_flashcards"
    )

    load_topics_button.click(
        fn=ui_populate_topic_dropdown,
        inputs=[], # No direct inputs, uses global kb
        outputs=[topic_dropdown_stored, status_display_stored] # Update dropdown and status
    )

    load_flashcards_button.click(
        fn=ui_display_stored_flashcards,
        inputs=[topic_dropdown_stored], # Pass selected topic
        outputs=[flashcards_display_stored, status_display_stored]
    )

    # Attempt to populate dropdown on load - this might run when the script is parsed by Gradio
    # iface.load(ui_populate_topic_dropdown, outputs=[topic_dropdown_stored, status_display_stored])
    # The .load() event is preferable but can be tricky with initial state.
    # For now, user clicks "Refresh/Load Topics".

    gr.Markdown(
        "<hr><p style='text-align:center; font-size:small;'>"
        "KompowLearn | AI-Powered Learning Tool | Remember to verify critical information."
        "</p>"
    )

if __name__ == "__main__":
    if initialization_error:
        print("\n--- GRADIO APP LAUNCH WARNING ---")
        print(f"The application is launching, but there was an initialization error(s):")
        print(initialization_error)
        print("The UI will load, but some functionalities might be impaired or fail, showing errors in the UI.")

    print("Launching Gradio UI... Check the console for the local URL (e.g., http://127.0.0.1:7860).")
    iface.launch()
    print("Gradio UI has been launched and should be accessible in your browser.")
