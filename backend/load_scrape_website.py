from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchResults
from .clean_data import clean_text


def search_duckduckgo(query: str) -> list[dict]:
    """
    Search DuckDuckGo for a given query and return a list of result of
    dictionaries containing title, link and snipppet.

    Args:
        query (str): The search query.

    Returns:
        list[dict]: A list from the search results .
    """
    search = DuckDuckGoSearchResults(output_format='list')
    results = search.invoke(query)
    return results[:3]


def load_website_content(link: str) -> list[Document]:
    """
    download website content from a url.

    Args:
        query (str): The link of the website to load.

    Returns:
        list[Document]:
        A list of LangChain Document objects containing page content.
    """
    docs = WebBaseLoader(link).load()
    return docs


def get_reduced_text(doc: Document) -> str:
    """
    Extracts the main part  of the document's text.

    Args:
        doc (Document): The original document.

    Returns:
        str: The reduced text that is clean.
    """
    full_text = doc.page_content
    half_length = int(len(full_text) * 0.3)
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    all_chunks = []
    for doc in docs:
        try:
            reduced_text = get_reduced_text(doc)
            chunks = splitter.split_text(reduced_text)

            for chunk in chunks:
                all_chunks.append(
                    Document(
                        page_content=chunk,
                        metadata=doc.metadata))
        except Exception as e:
            print(f"Error splitting document: {e}")

    return all_chunks
