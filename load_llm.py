import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


def load_llm()-> None:
    """
    Loads and returns a Google Generative AI language model (LLM) instance.

    This function checks if the 'GOOGLE_API_KEY' environment variable is set. If not,
    it prompts the user securely to input their Google AI API key using getpass and sets it
    in the environment variables.

    Returns:
        ChatGoogleGenerativeAI: An instance of the Gemini 2.0 Flash model configured with
        default settings (temperature=0, no token limit, 2 retries).
    """
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=120,
        max_retries=2,
    )
    return llm