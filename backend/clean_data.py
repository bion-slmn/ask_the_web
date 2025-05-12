import re


def merge_hyphenated_words(text: str) -> str:
    """
    Merges words that have been split across lines using a
    hyphen followed by a newline.

    For example, 'multi-\nline' becomes 'multiline'.

    Args:
        text (str): The input text containing hyphenated line breaks.

    Returns:
        str: Text with hyphenated line breaks merged into single words.
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def fix_newlines(text: str) -> str:
    """
    Replaces single newline characters with spaces to improve sentence flow,
    while preserving paragraph breaks (double newlines or more).

    Args:
        text (str): The input text with line breaks.

    Returns:
        str: Text with single newlines replaced by spaces.
    """
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text: str) -> str:
    """
    Reduces multiple consecutive newline characters to a single newline
    to normalize paragraph spacing.

    Args:
        text (str): The input text with excessive newlines.

    Returns:
        str: Text with multiple newlines reduced to a single newline.
    """
    return re.sub(r"\n{2,}", "\n", text)


def clean_text(text: str) -> str:
    """
    Cleans the input text by applying a series of formatting fixes:
    - Merges hyphenated words split by line breaks.
    - Replaces single newlines with spaces.
    - Reduces multiple newlines to a single one.

    Args:
        text (str): Text to be cleaned.

    Returns:
        str: Cleaned and normalized text.
    """
    cleaning_functions = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines]
    for cleaning_function in cleaning_functions:
        text = cleaning_function(text)
    return text
