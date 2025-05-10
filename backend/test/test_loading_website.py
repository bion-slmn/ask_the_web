import pytest
from langchain.schema import Document
from load_scrape_website import (
    search_duckduckgo,
    get_reduced_text,
    split_content
)

# --- Search Function ---

def test_search_duckduckgo_returns_links():
    query = "OpenAI ChatGPT"
    links = search_duckduckgo(query)
    assert isinstance(links, list)
    assert all(isinstance(link, str) for link in links)
    assert len(links) <= 2


# --- Mocking the Loader ---

@pytest.fixture
def dummy_doc():
    return Document(
        page_content="This is a test document.\n" * 100,
        metadata={"source": "test"}
    )


def test_get_reduced_text(dummy_doc):
    reduced = get_reduced_text(dummy_doc)
    assert isinstance(reduced, str)
    assert len(reduced) <= len(dummy_doc.page_content)
    assert "test document" in reduced


def test_split_content_returns_chunks(dummy_doc):
    docs = [dummy_doc]
    chunks = split_content(docs)
    assert isinstance(chunks, list)
    assert all(isinstance(c, Document) for c in chunks)
    assert all("test document" in c.page_content for c in chunks)
    assert all("source" in c.metadata for c in chunks)


def test_split_content_handles_empty_text():
    empty_doc = Document(page_content="", metadata={})
    chunks = split_content([empty_doc])
    assert chunks == []
