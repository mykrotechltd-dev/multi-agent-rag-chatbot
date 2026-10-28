"""Shared Langfuse environment configuration."""

import os
from typing import Optional

from dotenv import load_dotenv


def configure_langfuse_environment() -> Optional[str]:
    """Load Langfuse settings and map the legacy host variable to base URL.

    Langfuse's current SDKs use ``LANGFUSE_BASE_URL``. Older deployments of
    this application may still provide ``LANGFUSE_HOST``, so keep that name as
    a fallback while ensuring integrations such as LangChain see the current
    variable.
    """
    load_dotenv()
    base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    if base_url:
        base_url = base_url.rstrip("/")
        os.environ["LANGFUSE_BASE_URL"] = base_url
    return base_url
