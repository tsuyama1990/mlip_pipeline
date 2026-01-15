# ruff: noqa: D101, D103
"""Utilities for improving application resilience, such as retry mechanisms."""

import logging
import time
from functools import wraps
from typing import Any, Callable, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Callable[..., Any])


def retry(
    max_retries: int,
    delay_seconds: float = 1.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[T], T]:
    """Retry a function call on failure.

    This decorator will attempt to run the decorated function a total of
    `max_retries + 1` times (1 initial attempt + `max_retries` retries).

    Args:
        max_retries: Maximum number of retry attempts.
        delay_seconds: Time to wait between retries.
        exceptions: A tuple of exception types to catch and trigger a retry.

    Returns:
        The wrapped function.

    """

    def decorator(func: T) -> T:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            # The loop runs `max_retries + 1` times in total.
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        log_msg = (
                            "Attempt %d/%d for %s failed with %s. Retrying in %.2f s..."
                        )
                        logger.warning(
                            log_msg,
                            attempt + 1,
                            max_retries + 1,
                            func.__name__,
                            e,
                            delay_seconds,
                        )
                        time.sleep(delay_seconds)

            logger.error(
                "Function %s failed after %d retries. Giving up.",
                func.__name__,
                max_retries,
            )
            raise last_exception  # type: ignore

        return wrapper  # type: ignore

    return decorator
