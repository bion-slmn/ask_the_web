from typing import Optional
from graph import generate_graph
from dotenv import load_dotenv
import os
from rich.console import Console

# Load environment variables
load_dotenv()

os.environ['USER_AGENT'] = os.getenv(
    'USER_AGENT')

console = Console()


def ask_the_web(query: str) -> None:
    """
    Ask a question to the graph-based QA system and
    display the markdown-formatted response.

    Args:
        query (str): The question to be answered.

    Returns:

    """
    graph = generate_graph()
    content = ""
    usage_metadata = {}
    for chunk in graph.stream({"question": query}, stream_mode='values'):
        answer = chunk.get('answer')
        if answer and hasattr(answer, 'content'):
            content = answer.content
            usage_metadata = answer.usage_metadata

    return content, usage_metadata, chunk.get('status')
