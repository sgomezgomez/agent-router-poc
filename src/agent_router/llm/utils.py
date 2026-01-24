"""Utility functions for LLM service."""

import functools
import asyncio
import logging
from typing import Callable, Awaitable
from agent_router.core.types import JsonValue
from agent_router.core.errors import LLMProviderError

logger = logging.getLogger(__name__)


def clean_encoded_text(text: str) -> str:
    """Clean encoded text by removing common encoding artifacts.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return text

    # Remove common encoding artifacts
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2026': '...', # Ellipsis
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def calculate_retry_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float
) -> float:
    """Calculate retry delay with exponential backoff."""
    return min(base_delay * (exponential_base ** attempt), max_delay)


def create_retry_decorator(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0
) -> Callable:
    """Create a retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff calculation

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., Awaitable[JsonValue]]) -> Callable[..., Awaitable[JsonValue]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> JsonValue:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff
                        delay = calculate_retry_delay(
                            attempt,
                            base_delay,
                            max_delay,
                            exponential_base
                        )
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts: {str(e)}"
                        )

            # All retries exhausted
            raise LLMProviderError(
                f"Failed after {max_retries} attempts: {str(last_exception)}"
            )

        return wrapper
    return decorator
