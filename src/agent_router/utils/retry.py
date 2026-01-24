"""Retry decorator for async functions."""

import asyncio
from functools import wraps
from typing import TypeVar, Callable

T = TypeVar("T")


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Decorator for async retry with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry

    Example:
        @async_retry(max_attempts=3, delay=1.0, backoff=2.0)
        async def my_function():
            # Function that might fail
            pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        raise

                    # Log retry attempt if logger is available
                    logger = kwargs.get("logger")
                    if logger:
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed, retrying in {current_delay}s",
                            function=func.__name__,
                            error=str(e),
                        )

                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            raise last_exception  # type: ignore

        return wrapper

    return decorator
