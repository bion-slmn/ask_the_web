from nodes import (State, get_links, scrape_web_data, 
                   generate_answer, send_to_scrape_data,
                     verify_citations)
from langgraph.graph import START, StateGraph


def generate_graph() -> StateGraph:
    """
    Builds and compiles a LangGraph state machine for a web-based question-answering workflow.

    The graph includes the following steps:
        1. `get_links` – Identifies and retrieves relevant web page links based on the user query.
        2. `scrape_web_data` – Scrapes the contents of the selected links for contextual information.
        3. `generate_answer` – Uses a language model to generate an answer from the scraped content.
        4. `verify_citations` – Verifies whether the citations in the generated answer are genuinely supported by the source content.

    The flow of the graph is:
        START → get_links → scrape_web_data → generate_answer → verify_citations

    The transition from `get_links` to `scrape_web_data` is determined by the `send_to_scrape_data` condition.

    Returns:
        StateGraph: A compiled LangGraph StateGraph object that defines the complete processing workflow.
    """
    graph_builder = StateGraph(State)
    graph_builder.add_node(get_links)
    graph_builder.add_node(scrape_web_data)
    graph_builder.add_node(generate_answer)
    graph_builder.add_node(verify_citations)


    graph_builder.add_edge(START, "get_links")
    graph_builder.add_conditional_edges('get_links', send_to_scrape_data, ['scrape_web_data'])
    graph_builder.add_edge('scrape_web_data', 'generate_answer')
    graph_builder.add_edge('generate_answer', 'verify_citations')
    graph = graph_builder.compile()
    return graph
