import pytest
from langchain.schema import Document
from unittest.mock import patch
from ..load_scrape_website import (
    search_duckduckgo,
    get_reduced_text,
    split_content
)


@patch("backend.load_scrape_website.DuckDuckGoSearchResults")
def test_search_duckduckgo_returns_links(mock_search: patch) -> None:
    """
    Test that search_duckduckgo returns a list of up to 3 string URLs.
    """
    mock_search_instance = mock_search.return_value
    mock_search_instance.invoke.return_value = [
        "https://example.com",
        "https://openai.com",
        "https://chat.openai.com"
    ]
    query = "OpenAI ChatGPT"
    links = search_duckduckgo(query)
    assert isinstance(links, list)
    assert all(isinstance(link, str) for link in links)
    assert len(links) <= 3


# --- Fixtures ---

@pytest.fixture
def dummy_doc() -> Document:
    """
    Fixture providing a dummy Document with repeated content.
    """
    return Document(
        page_content="This is a test document.\n" * 100,
        metadata={"source": "test"}
    )


# --- Test: get_reduced_text ---

@patch("backend.load_scrape_website.clean_text")
def test_get_reduced_text_calls_clean_text(
        mock_clean: patch,
        dummy_doc: Document) -> None:
    """
    Test that get_reduced_text calls clean_text and returns the cleaned result.
    """
    mock_clean.return_value = "Cleaned text"
    result = get_reduced_text(dummy_doc)
    assert result == "Cleaned text"
    mock_clean.assert_called_once()


# --- Test: split_content ---

@patch("backend.load_scrape_website.get_reduced_text")
def test_split_content_returns_chunks(
        mock_reduce: patch,
        dummy_doc: Document) -> None:
    """
    Test that split_content returns chunked Documents with proper metadata.
    """
    mock_reduce.return_value = "This is a test document. " * 20  # ~500 chars
    chunks = split_content([dummy_doc])
    assert isinstance(chunks, list)
    assert all(isinstance(c, Document) for c in chunks)
    assert all("test document" in c.page_content for c in chunks)
    assert all("source" in c.metadata for c in chunks)


@patch("backend.load_scrape_website.get_reduced_text")
def test_split_content_handles_empty_text(mock_reduce: patch) -> None:
    """
    Test that split_content handles empty text gracefully and
    returns an empty list.
    """
    mock_reduce.return_value = ""
    empty_doc = Document(page_content="", metadata={})
    chunks = split_content([empty_doc])
    assert chunks == []


@patch("backend.load_scrape_website.get_reduced_text",
       side_effect=Exception("Split error"))
def test_split_content_handles_exceptions(
        mock_reduce: patch,
        dummy_doc: Document,
        capsys: pytest.CaptureFixture) -> None:
    """
    Test that split_content catches and logs exceptions during text splitting.
    """
    chunks = split_content([dummy_doc])
    assert chunks == []
    captured = capsys.readouterr()
    assert "Error splitting document" in captured.out
