from typing import List, Annotated, Tuple
from langchain_core.documents import Document
from typing_extensions import TypedDict
from load_llm import load_llm
from load_scrape_website import load_website_content, split_content
from prompts import GENERATE_RESULT_PROMPT, VERIFY_PROMPT


# Load the language model and prompt template
llm = load_llm()

class CitationStatus(TypedDict):
    """
    A TypedDict to represent the verification status of a citation in an answer.

    Attributes:
        status (str): Indicates whether the citation passed or failed verification.
                      Expected values are 'PASS' or 'FAIL'.
    """
    status: str


class AnswerWithSources(TypedDict):
    """
    A structured answer to the question with cited sources.

    Attributes:
        answer (str): The generated answer to the user's question.
        sources (List[Tuple[str, str]]): A list of (title, link) pairs representing cited sources.
    """
    answer: str
    sources: Annotated[
        List[Tuple[str, str]],
        "List of sources (title, link) used to answer the question",
    ]


class State(TypedDict):
    """
    The full state of the question-answering graph.

    Attributes:
        question (str): The user's input question.
        context (List[Document]): The documents retrieved to answer the question.
        answer (AnswerWithSources): The final answer including source citations.
    """
    question: str
    context: List[Document]
    answer: AnswerWithSources
    status: CitationStatus


def retrieve(state: State) -> dict:
    """
    Retrieve relevant documents based on the question.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: A dictionary containing the retrieved context documents.
    """
    docs = load_website_content(state["question"])
    retrieved_docs = split_content(docs)
    return {"context": retrieved_docs}


def generate(state: State) -> dict:
    """
    Generate an answer using the language model and retrieved context.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: A dictionary containing the generated answer and sources.
    """
    formatted_prompt = GENERATE_RESULT_PROMPT.format(question=state["question"], context=state['context'])
    response = llm.invoke(formatted_prompt)
    return {"answer": response}


def verify_citations(state: State) -> dict:
    """
    Verifies whether the citations in a given answer are genuinely supported 
    by the associated content.

    This function formats a verification prompt using the provided answer and 
    its related context, then invokes a language model that returns a structured 
    result conforming to the CitationStatus TypedDict.

    Returns:
        dict: A dictionary with a single key:
            - 'status' (str): Either 'PASS' or 'FAIL', indicating whether the 
              citations were deemed appropriate according to the source content.
    """
    prompt = VERIFY_PROMPT.format(citations=state['answer'], content=state['context'])
    structured_llm = llm.with_structured_output(CitationStatus)
    response = structured_llm.invoke(prompt)
    print(response)
    print('--' * 20)
    return {"status": response["status"]}
