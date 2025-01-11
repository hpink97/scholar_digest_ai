# services/llm.py

import os
import dotenv
import streamlit as st
from openai import OpenAI

dotenv.load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
if len(OPENROUTER_KEY) == 0:
    st.error("API key not found. Please provide an Open Router API key to use this app.")


class ScholarDigestAI:
    def __init__(self, api_key=OPENROUTER_KEY, article_text=None):
        """
        Initialize the ScholarDigestAI object.

        :param api_key: API key for OpenAI or equivalent LLM provider.
        :param article_text: Optional text from a single paper if only one DOI is loaded.
        """
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API key must be provided or set in environment variables.")

        self.base_url = "https://openrouter.ai/api/v1"
        self.client = self._init_client()
        
        self.delimiter = "\n\n"
        self.article_text = article_text
        self.system_prompt = self._generate_system_prompt()
        self.conversation = [self.system_prompt]

        # Dictionaries or other structures to support custom formatting and prompts
        self.FORMAT_TO_PROMPT = {
            "TL;DR": (
                "Please provide a TL;DR summary. Be very concise, focus on ONLY the key results and their wider context. "
                "Avoid details, keep it under 50 words if possible. No jargon or acronyms."
                "Only return the TL;DR summary, no additional information or bullet points."
            ),
            "Concise Bullet Points": (
                "Please structure your response in concise bullet points. "
                "Limit each bullet to a key finding or insight."
            ),
            "Short summary": (
                "Provide a short summary, around 250-400 words, covering key insights "
                "without going into extensive detail. Focus on the main results and their implications."
            ),
            "Detailed summary": (
                "Provide a comprehensive, detailed summary. You may include sections like; "
                "Short Background, Hypothesis, Key Results, Impact (and optionally, Limitations). "
                "No need to include minor limitations for impressive papers. "


            ),
        }

        self.TECHNICAL_LEVEL_PROMPTS = {
            "elementary": (
                "Explain this as if I'm about 5 years old. "
                "Use extremely simple language and familiar, everyday examples. "
                "Avoid all jargon and acronyms."
            ),
            "high school": (
                "Explain this as if I've completed high-school-level science. "
                "I know basic biology, chemistry, or physics terms. Provide some analogies "
                "but keep them straightforward. Avoid acronyms or jargon."
                "Any specialized terms should be clearly explained."
            ),
            "non-specialist": (
                "Explain this to a lay audience or member of the general public with no specific scientific background. "
                "Focus on the why a member of the general public should care? What are the an wider implications of the paper" 
                "in everyday life. "
                "Avoid jargon. Avoid specalist language. Avoid acronyms. Don't assume any prior knowledge of the field."
            ),
            "undergrad": (
                "Explain this assuming I am an undergrad student in the broad field. "
                "I am not fimilar with advanced concepts, jargon or acronyms in this niche field. "
                "Use simple language, explain concepts, and provide context."
                "I am looking to understand the main concepts, and wider significance (not any of the nitty-gritty)."
            ),
            "domain expert": (
                "Explain this assuming I am well-versed in the field. "
                "Feel free to include advanced concepts, nuanced references, and "
                "comparisons to other literature. Still avoid acroyms, or define if you do use them."
                "Emphasize new breakthroughs, methodological details, and any limitations."
            ),
        }

    
    def _init_client(self):
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        return client
        
    def reset_conversation(self):
        """Reset the conversation history to just the system prompt."""
        self.conversation = [self.system_prompt]

    def _generate_system_prompt(self):
        """
        Build a base system prompt with richer context and instructions.

        This prompt is used in all chat completions.
        """
        instructions = (
            "You are ScholarDigestAI, an AI assistant specialised in summarizing and explaining academic papers. "
            "Your top priority is to present scientific content in an accessible, accurate, and jargon-free manner "
            "to maximise understand key concepts without needing to become a domain specialist. "
            "You must:\n"
            "1. Avoid excessive jargon, and define or explain any specialized terms or acronyms.\n"
            "2. Provide context for why findings are significant \n"
            "3. Keep answers as clear and concise as possible, ensure all text aids understanding of the paper.\n"
            "4. Whenever referencing the background text or relevant sections, present them in a coherent, easy-to-follow manner.\n"
            "5. Strive to highlight real-world implications or broader impacts if they are relevant.\n"
            "By following these principles, ensure that any user—regardless of their expertise—can grasp the core insights of the paper. "
        )

        background = f"Here is some relevant text:\n{self.article_text}" if self.article_text else ""
        return {
            "role": "system",
            "content": f"{instructions}{self.delimiter}{background}{self.delimiter}",
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

    def _build_formatting_instructions(self, format_choice: str, language: str) -> str:
        """
        Returns extra prompt instructions based on the user's format and language choices.
        """
        # Grab the formatting text from the dictionary, with a fallback
        format_instructions = self.FORMAT_TO_PROMPT.get(
            format_choice, 
            "Please provide a concise summary."
        )

        # Incorporate the language choice
        language_instructions = (
            f"Respond in {language}. If the paper is in another language, translate or summarize as needed."
        )

        # Combine them as you see fit
        final_instructions = (
            f"{format_instructions}{self.delimiter}"
            f"{language_instructions}{self.delimiter}"
        )
        return final_instructions

    def _format_relevant_sections(self, relevant_sections) -> str:
        """
        Combine or format relevant sections from embeddings, references, or other sources.
        This method allows you to unify how multiple text snippets are appended to the prompt.

        :param relevant_sections: A list (or single str) representing paper snippets or references.
        :return: A formatted string ready to be added to the final prompt.
        """
        if not relevant_sections:
            return ""

        # If it's already a single string, just return it
        if isinstance(relevant_sections, list):
            combined_text = self.delimiter.join(relevant_sections)
        else:
            combined_text = relevant_sections

        
        return (
            f"Here is the relevant section of the paper you can use to answer the question (if applicable):{self.delimiter}"
            f"{combined_text}{self.delimiter}"
        )
    
    def _format_techinical_level(self, technical_level: str) -> str:
        """
        Returns extra prompt instructions based on the user's technical level choice.
        """
        # Grab the technical level text from the dictionary, with a fallback
        technical_level_instructions = self.TECHNICAL_LEVEL_PROMPTS.get(
            technical_level, 
            "Please avoid jargon, acroymns or advanced concepts and provide a clear explanation."
        )

        return (
            f"{technical_level_instructions}{self.delimiter}"
        )

    def ask_question(
        self,
        question: str,
        model: str,
        technical_level: str,
        format: str,
        language: str,
        relevant_sections=None,
    ) -> str:
        """
        Ask a question and get a response from the LLM.
        
        :param question: The user's question or request.
        :param model: LLM model to be used (e.g., "google/gemini-flash-1.5").
        :param technical_level: The desired technical level (e.g., "elementary").
        :param format: The chosen response format (e.g., "TL;DR", "Detailed summary").
        :param language: The target language for the response (e.g., "English", "Spanish").
        :param relevant_sections: Optional list or string of text from embeddings search or references.
        :return: The LLM's answer as a string.
        """


        # Create the user message
        user_prompt = (
            f"{question}\n\n"
            f"{self._format_relevant_sections(relevant_sections)}"
            f"{self._format_techinical_level(technical_level)}"
            f"{self._build_formatting_instructions(format, language)}"
        )

        user_message = {"role": "user", "content": user_prompt}
        self.conversation.append(user_message)

        # 4) Call the LLM
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=self.conversation,
            )
            # 5) Parse and return the LLM response
            return self._parse_chat_response(completion)
        except Exception as e:
            return f"Error: {e}"
