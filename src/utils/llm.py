"""Utility functions for LLM initialization."""

from typing import Optional

from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from utils.langfuse_config import configure_langfuse_environment
from utils.model_provider import get_model_api_config


def initialize_llm(
    model: str,
    langfuse_handler: Optional[CallbackHandler] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ChatOpenAI:
    """
    Initialize a ChatOpenAI LLM instance with Langfuse instrumentation.
    
    Uses OpenRouter when OPENROUTER_API_KEY is configured, with legacy
    OpenAI environment variables supported as a fallback.
    
    Args:
        model: LLM model name (e.g., "gpt-4o-mini")
        langfuse_handler: Optional Langfuse callback handler. If None, creates a new one.
        temperature: Temperature for the LLM (default: 0.0)
        api_key: Optional API key. If None, reads from the provider environment.
        base_url: Optional base URL. If None, uses the configured provider URL.
    
    Returns:
        Initialized ChatOpenAI instance.
    
    Raises:
        ValueError: If no provider API key is configured.
    """
    api_key, base_url = get_model_api_config(api_key=api_key, base_url=base_url)

    if not api_key:
        raise ValueError(
            "No model API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY."
        )
    
    # Initialize Langfuse handler if not provided
    # CallbackHandler reads from environment variables automatically
    if langfuse_handler is None:
        configure_langfuse_environment()
        langfuse_handler = CallbackHandler()
    
    # ChatOpenAI uses OpenRouter's OpenAI-compatible API when base_url is set.
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        callbacks=[langfuse_handler],
        temperature=temperature,
    )

