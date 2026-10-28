"""Shared configuration for OpenRouter and OpenAI-compatible model APIs."""

import os
from typing import Optional, Tuple


OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def get_model_api_config(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return credentials for the configured provider, preferring OpenRouter.

    OPENAI_API_KEY and OPENAI_API_BASE remain supported for existing
    OpenAI deployments and local setups.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if api_key is None:
        api_key = openrouter_key or os.getenv("OPENAI_API_KEY")

    if base_url is None:
        if openrouter_key and api_key == openrouter_key:
            base_url = os.getenv("OPENROUTER_API_BASE") or OPENROUTER_DEFAULT_BASE_URL
        else:
            base_url = os.getenv("OPENROUTER_API_BASE") or os.getenv("OPENAI_API_BASE")

    return api_key, base_url
