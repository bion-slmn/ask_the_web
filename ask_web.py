from typing import Optional
from graph import generate_graph
from dotenv import load_dotenv
import os
from rich.console import Console
from rich.markdown import Markdown

# Load environment variables
load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ['USER_AGENT'] = os.getenv('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/')

console = Console()

def ask_the_web(query: str) -> None:
    """
    Ask a question to the graph-based QA system and display the markdown-formatted response.

    Args:
        query (str): The question to be answered.

    Returns:
        None
    """
    content: Optional[str] = None
    graph = generate_graph()

    try:
        for chunk in graph.stream({"question": query}, stream_mode='values'):
            answer = chunk.get('answer')
            if answer and hasattr(answer, 'content'):
                content = answer.content
                console.log(answer.usage_metadata)  # Using Rich's logger
    except Exception as e:
        console.print(f"[bold red]Error during graph streaming:[/bold red] {e}")
        return

    if content:
        console.print(Markdown(content))  # Render markdown nicely
    else:
        console.print("[yellow]No content found.[/yellow]")


ask_the_web("tell me about france?")