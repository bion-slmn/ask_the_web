from nodes import generate, retrieve, verify_citations, State
from langgraph.graph import START, StateGraph


def generate_graph() -> StateGraph:
    """
    Builds and compiles a LangGraph state machine for a question-answering workflow.

    This graph consists of three sequential steps:
        1. `retrieve` – Fetches relevant documents or source content.
        2. `generate` – Uses a language model to generate an answer based on the retrieved context.
        3. `verify_citations`  Verifies that the generated answer's citations are supported by the source material.

    The graph begins at the START node and proceeds through each step in order.

    Returns:
        StateGraph: A compiled StateGraph object representing the defined workflow.
    """
    graph_builder = StateGraph(State).add_sequence([retrieve, generate, verify_citations])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph
