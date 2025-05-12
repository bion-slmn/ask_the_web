import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


def load_llm() -> None:
    """
    Loads and returns a Google Generative AI language model (LLM) instance.

    Returns:
        ChatGoogleGenerativeAI: An instance of the Gemini 2.0 Flash model
    """
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass(
            "Enter your Google AI API key: ")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=120,
        max_retries=2,
    )
    return llm
