import os
import dotenv
import streamlit as st
from openai import OpenAI


# Load .env for local development
if not os.getenv("STREAMLIT_CLOUD"):  # Check if running on Streamlit Cloud
    dotenv.load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", st.secrets.get("OPENROUTER_API_KEY"))



class ScholarDigestAI:
    def __init__(self, api_key=OPENROUTER_KEY, article_text=None):
        """
        Initialize the ScholarDigestAI object.

        :param api_key: API key for OpenAI or equivalent LLM provider.
        :param model: Default model to use for completions.
        """
        self.api_key = api_key 
        if not self.api_key:
            raise ValueError("API key must be provided or set in environment variables.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.article_text = article_text
        self.system_prompt = self._generate_system_prompt()
        self.conversation = [self.system_prompt]

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation = [self.system_prompt]

    def _generate_system_prompt(self) -> str:
        """
        Build a system prompt with context and instructions.

        :param technical_level: User's requested technical level.
        :param article_text: Relevant text from the article, if any.
        :return: A formatted system message string.
        """
        instructions = (
            "You are ScholarDigestAI, tasked with summarizing and explaining academic papers. "
            "Your goal is to be as clear as possible while matching the user’s technical level. "
            "Avoid excessive jargon and provide context to help with understanding."
        )
        background = f"Here is some relevant text:\n{self.article_text}" if self.article_text else ""
        return {
            "role": "system",
            "content": f"{instructions}\n\n{background}\n",
        }

    def _parse_chat_response(self, completion):
        """
        Parse the response from the LLM chat completion.

        :param completion: Completion object from the LLM chat API.
        :return: The LLM's response as a string.
        """
        if completion.choices:
            response = completion.choices[0].message.content
            # Add assistant response to conversation history
            self.conversation.append({"role": "assistant", "content": response})
            return response
        elif completion.error:
            return f"Error from LLM: {completion.error}"

        return "Error: Unknown LLM response."

    def ask_question(
        self,
        question: str,
        model: str,
        technical_level: str,
        relevant_sections=None,
    ) -> str:
        """
        Ask a question and get a response from the LLM.
        """

        # Handle article text or search function for context
        if relevant_sections is not None:
            if isinstance(relevant_sections, list):
                relevant_sections = "\n\n".join(relevant_sections)

            background = (
                "Here is the relevant section of the paper you can use to answer the question (if applicable):\n"
                f"{relevant_sections}\n\n"
            )
        else:
            background = ""

        # Add user s
        user_message = {
            "role": "user",
            "content": f"{question}\n\n{background}Please give your answer at a {technical_level} level.",
        }
        self.conversation.append(user_message)

        # Call the LLM
        completion = self.client.chat.completions.create(
            model=model,
            messages=[self.system_prompt] + self.conversation,
        )

        return self._parse_chat_response(completion)
