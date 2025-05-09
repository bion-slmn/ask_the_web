from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchResults


api_wrapper = DuckDuckGoSearchAPIWrapper(
    region="us-en",     # or "de-de", etc.
    time="d",           # results from the last day
    max_results=2
)

search = DuckDuckGoSearchResults(api_wrapper=api_wrapper, output_format='list')

def search_duckduckgo(query: str) -> list[str]:
    """
    Search DuckDuckGo for a given query and return a list of result URLs.

    Args:
        query (str): The search query.

    Returns:
        list[str]: A list of URLs from the search results.
    """
    results = search.invoke(query)
    links = [link['link'] for link in results if 'link' in link]
    return links[:2]


def load_website_content(query: str) -> list[Document]:
    """
    Load website content from URLs returned by a DuckDuckGo search.

    Args:
        query (str): The search query.

    Returns:
        list[Document]: A list of LangChain Document objects containing page content.
    """
    links = search_duckduckgo(query)
    if not links:
        raise ValueError("No links found from DuckDuckGo search.")

    docs = WebBaseLoader(links).load()
    return docs


def split_content(docs: list[Document]) -> list[Document]:
    """
    Split a list of Document objects into smaller chunks for processing.

    Args:
        docs (list[Document]): A list of LangChain Document objects.

    Returns:
        list[Document]: A list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    for doc in docs:
        try:
            full_text = doc.page_content
            half_lenght = (len(full_text)) // 2 
            reduced_text = full_text[:half_lenght]   
            chunks = splitter.split_text(reduced_text)
            for chunk in chunks:
                all_chunks.append(Document(page_content=chunk, metadata=doc.metadata))
        except Exception as e:
            print(f"Error splitting document: {e}")

    return all_chunks
