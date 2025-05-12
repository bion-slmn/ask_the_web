from .nodes import (State, get_links, scrape_web_data,
                    generate_answer, send_to_scrape_data,
                    verify_citations)
from langgraph.graph import START, StateGraph
import os
from dotenv import load_dotenv

load_dotenv()


def generate_graph() -> StateGraph:
    """
    Builds and compiles a LangGraph state machine
    for a web-based question-answering workflow.

    The flow of the graph is:
    START → get_links → scrape_web_data → generate_answer → verify_citations

    The transition from `get_links` to `scrape_web_data` is
    determined by the `send_to_scrape_data` condition.

    Returns:
        StateGraph: A compiled LangGraph StateGraph object
    """
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Web Assistant QA"

    graph_builder = StateGraph(State)
    graph_builder.add_node(get_links)
    graph_builder.add_node(scrape_web_data)
    graph_builder.add_node(generate_answer)
    graph_builder.add_node(verify_citations)

    graph_builder.add_edge(START, "get_links")
    graph_builder.add_conditional_edges(
        'get_links',
        send_to_scrape_data,
        ['scrape_web_data']
    )
    graph_builder.add_edge('scrape_web_data', 'generate_answer')
    graph_builder.add_edge('generate_answer', 'verify_citations')
    graph = graph_builder.compile()

    return graph
