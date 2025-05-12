GENERATE_RESULT_PROMPT = '''
You are an informative assistant. Use the only provided context to clearly answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Cite sources using square brackets with numbers (e.g., [1], [2]). At the end, include a "Sources" section
 listing each number, title, and URL. Numbering should always start from 1

Example:

Question: What are the main features of LangChain?

Answer:
LangChain is a framework with tools for language model apps, such as chains, prompts, and agents [1].
 It's used for QA, chatbots, and summarization [2].

Sources:
[1] LangChain Docs, https://docs.langchain.com/
[2] LangChain Use Cases, https://langchain.com/use-cases/

Question: {question}

Context:
{context}
'''

VERIFY_PROMPT = '''
Given the following answer with citations and the corresponding source snippets,
determine whether each citation genuinely supports the sentence it is attached to.

Use the `load_content_from_link` function to retrieve source text based on the citation's URL.
Then verify that each sentence is supported by the actual source content.
if no citation or source is found, return "None"

Respond with "PASS" if the citation supports the sentence, otherwise "FAIL" .

Answer:
{citations}

Sources:
{content}
'''
