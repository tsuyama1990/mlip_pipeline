"""
This module provides generic, reusable patterns for building robust functions,
such as retry decorators.
"""
import logging
import time
from functools import wraps
from typing import Any, Callable, Type

logger = logging.getLogger(__name__)


def retry(
    attempts: int,
    delay: float,
    exceptions: tuple[Type[Exception], ...],
    on_retry: Callable | None = None,
) -> Callable:
    """
    A decorator that retries a function, with an optional callback to modify
    arguments on failure.

    Args:
        attempts: The maximum number of times to try the function.
        delay: The number of seconds to wait between retries.
        exceptions: A tuple of exception types to catch and trigger a retry.
        on_retry: An optional function to call before a retry. It receives the
                  exception and the keyword arguments of the failed call. It
                  should return a dictionary of updated keyword arguments.

    Returns:
        A decorator that can be applied to a function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_kwargs = kwargs.copy()
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **current_kwargs)
                except exceptions as e:
                    if attempt == attempts:
                        logger.error(
                            "Function %s failed after %d attempts.",
                            func.__name__,
                            attempts,
                            exc_info=True,
                        )
                        raise
                    logger.warning(
                        "Attempt %d/%d for %s failed with %s.",
                        attempt,
                        attempts,
                        func.__name__,
                        e.__class__.__name__,
                    )
                    if on_retry:
                        new_kwargs = on_retry(e, current_kwargs)
                        if new_kwargs:
                            logger.info("Retrying with modified parameters...")
                            current_kwargs.update(new_kwargs)
                        else:
                            logger.info("No parameter modifications suggested by on_retry handler.")

                    logger.info("Retrying in %.2f seconds...", delay)
                    time.sleep(delay)

        return wrapper

    return decorator
