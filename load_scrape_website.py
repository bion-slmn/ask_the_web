from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchResults
from clean_data import clean_text


api_wrapper = DuckDuckGoSearchAPIWrapper(
    region="us-en",     # or "de-de", etc.
    time="d",           # results from the last day
    max_results=2
)



def search_duckduckgo(query: str) -> list[str]:
    """
    Search DuckDuckGo for a given query and return a list of result URLs.

    Args:
        query (str): The search query.

    Returns:
        list[str]: A list of URLs from the search results.
    """
    search = DuckDuckGoSearchResults(api_wrapper=api_wrapper, output_format='list')
    results = search.invoke(query)
    links = [link['link'] for link in results if 'link' in link]
    return links[:2]


def load_website_content(link: str) -> list[Document]:
    """
    Load website content from URLs returned by a DuckDuckGo search.

    Args:
        query (str): The link of the website to load.

    Returns:
        list[Document]: A list of LangChain Document objects containing page content.
    """
    docs = WebBaseLoader(link).load()
    return docs

def get_reduced_text(doc: Document) -> str:
    """
    Extracts the first half of the document's text.

    Args:
        doc (Document): The original document.

    Returns:
        str: The reduced text.
    """
    full_text = doc.page_content
    half_length = len(full_text) // 2
    reducecd_doc = full_text[:half_length]
    return clean_text(reducecd_doc)



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
            reduced_text = get_reduced_text(doc)    
            chunks = splitter.split_text(reduced_text)
            
            for chunk in chunks:
                all_chunks.append(Document(page_content=chunk, metadata=doc.metadata))
        except Exception as e:
            print(f"Error splitting document: {e}")

    return all_chunks
