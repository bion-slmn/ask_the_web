import streamlit as st
import time
import json
from typing import Tuple, Dict, Any
from update_telemery import update_telemetry
from backend.graph import generate_graph


def display_ui() -> str:
    """
    Display the Streamlit UI elements for user interaction.

    Returns:
        str: The user's input query string, or an empty string if no query was submitted.
    """
    st.title("📚 Ask the Web Assistant")
    query = st.text_input("Ask a question:")
    run_query = st.button("Ask")
    if run_query and query:
        return query
    return ""


def init_placeholders() -> Tuple[st.delta_generator.DeltaGenerator, st.delta_generator.DeltaGenerator, st.delta_generator.DeltaGenerator]:
    """
    Initialize the placeholder elements for displaying the answer, status, and debug information.

    Returns:
        Tuple: A tuple containing:
            - answer_box (DeltaGenerator): A placeholder for the answer.
            - status_box (DeltaGenerator): A placeholder for the status.
            - debug_container (DeltaGenerator): A container for expandable debug info.
    """
    answer_box = st.empty()
    status_box = st.empty()
    debug_container = st.container()
    return answer_box, status_box, debug_container


def process_query(
    query: str,
    graph: Any,
    answer_box: st.delta_generator.DeltaGenerator,
    status_box: st.delta_generator.DeltaGenerator,
    debug_container: st.delta_generator.DeltaGenerator
) -> Tuple[str, Dict[str, Any], str]:
    """
    Processes the user's query by streaming chunks from the backend graph and updating UI components.

    Args:
        query (str): The user's question.
        graph (Any): The graph object capable of streaming responses.
        answer_box (DeltaGenerator): Placeholder to display the answer.
        status_box (DeltaGenerator): Placeholder to display the status.
        debug_container (DeltaGenerator): Container to show expandable debug info.

    Returns:
        Tuple[str, dict, str]: The final answer text, usage metadata, and status string.
    """
    answer_text = ''
    usage_metadata = {}
    status = "No status available"
    debug_data = []  # collect all raw_results for debug info

    with st.spinner("🔄 Reading web, downloading and response"):
        for chunk in graph.stream({"question": query}, stream_mode='values'):
            if chunk.get('answer'):
                answer_text = chunk['answer'].content
                answer_box.markdown(f"### ✅ Answer\n{answer_text}")
                usage_metadata = chunk['answer'].usage_metadata

            if chunk.get('status'):
                status = chunk['status']
                status_box.markdown("### 📜 Status")
                status_box.markdown(status)

            if chunk.get('raw_results') and not debug_data:
                debug_data.append(chunk['raw_results'])

    # Show debug info after stream completes
    with debug_container.expander("🔍 Debug Info", expanded=False):
        st.markdown("#### Raw Search Results (JSON)")
        for i, data in enumerate(debug_data, 1):
            st.markdown(f"**Chunk {i}**")
            st.json(data)

    return answer_text, usage_metadata, status


def main() -> None:
    """
    The main entry point for the Streamlit app. Orchestrates UI setup, query processing, and telemetry updates.
    """
    query = display_ui()
    if query:
        start = time.time()
        graph = generate_graph()
        answer_box, status_box, debug_section = init_placeholders()

        try:
            answer_text, usage_metadata, status = process_query(query, graph, answer_box, status_box, debug_section)
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            return

        st.markdown("### 📜 Final Status")
        st.markdown(status)

        try:
            update_telemetry(start, usage_metadata)
        except Exception as e:
            st.warning(f"⚠️ Could not update telemetry: {e}")


if __name__ == "__main__":
    main()
