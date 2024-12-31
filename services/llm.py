# services/llm.py

import os
import dotenv

# Example of an OpenAI-like client
from openai import OpenAI

dotenv.load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Initialize client (OpenAI or your LLM provider)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)


def build_system_prompt(technical_level="high school", article_text=None) -> str:
    """
    Build a system (assistant) prompt providing context + instructions
    based on the user's technical level and, optionally, text from the article or search.
    """
    instructions = (
        "You are ScholarDigestAI, tasked with summarizing and explaining academic papers. "
        "Your goal is to be as clear as possible while matching the user’s technical level. "
        "Avoid excessive jargon and provide context to help with understanding."
    )
    background = f"Here is some relevant text:\n{article_text}" if article_text else ""
    return (
        f"{instructions}\n"
        f"Technical Level: {technical_level}\n\n"
        f"{background}"
    )


def get_llm_response(conversation, llm_model="google/gemini-2.0-flash-thinking-exp:free") -> str:
    """
    conversation: list of dicts, each with {"role": "system"/"user"/"assistant", "content": "..."}.
    Returns the LLM's reply text.
    """
    completion = client.chat.completions.create(
        model=llm_model,
        messages=conversation
    )
    if completion.choices:
        return completion.choices[0].message.content
    return "Error: No response from LLM."


def explain_paper(
    question: str,
    model: str,
    techincal_level: str,
    article_text: str = None,
    search_fn=None,
) -> str:
    """
    If article_text is passed, use it directly. Otherwise, do a semantic search
    (via search_fn) and retrieve relevant text chunks.
    """
    if article_text:
        background = f"Here is the article text: {article_text}"
    else:
        # If no article_text, we assume multiple DOIs => semantic search
        # `search_fn` is a callback to your search_database function
        relevant_sections = search_fn(question) if search_fn else ""
        background = (
            "Here is the relevant section of the paper you can use to answer the question (where applicable):\n"
            f"{relevant_sections}"
        )

    # Build conversation
    system_msg = {
        "role": "system",
        "content": (
            "You are a ScholarDigestAI tasked with summarising an academic paper. "
            "Your goal is to summarize and explain the answer in a clear and concise manner, "
            "minimizing the use of technical terms where possible. Provide context to assist in understanding "
            "rather than just repeating the text.\n"
            f"{background}"
        )
    }
    user_msg = {
        "role": "user",
        "content": f"{question}\nPlease give your answer at a {techincal_level} level."
    }
    conversation = [system_msg, user_msg]

    completion = client.chat.completions.create(
        model=model,
        messages=conversation,
    )

    if completion.choices:
        return completion.choices[0].message.content
    elif completion.error:
        return f"Error from LLM: {completion.error}"

    return "Error: Unknown LLM response."
