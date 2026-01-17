"""
This module provides generic, reusable patterns for building robust functions,
such as retry decorators.
"""
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


def retry(
    attempts: int,
    delay: float,
    exceptions: tuple[type[Exception], ...],
    on_retry: Callable | None = None,
) -> Callable:
    """
    A decorator that retries a function upon specific exceptions, with an
    optional callback to modify arguments for the next attempt.

    This decorator provides a powerful mechanism for building resilient functions.
    When the decorated function raises an exception listed in the `exceptions`
    tuple, the decorator will catch it and, instead of crashing, will wait for a
    specified `delay` and then re-invoke the function. This is repeated up to
    the number of `attempts`.

    A key feature is the `on_retry` callback, which allows for domain-specific
    error handling. Before a retry, this callback is invoked with the caught
    exception and the keyword arguments of the failed function call. The
    callback can inspect the error and return a dictionary of modified keyword
    arguments, which will be used in the next attempt. This enables intelligent
    recovery strategies, such as adjusting numerical parameters in response to
    a convergence failure.

    Args:
        attempts: The maximum number of times to try the function. Must be >= 1.
        delay: The number of seconds to wait between retries.
        exceptions: A tuple of exception types that should trigger a retry.
        on_retry: An optional function to call before a retry.
                  It receives two arguments:
                  - The exception instance that was caught.
                  - A dictionary of the keyword arguments from the failed call.
                  It should return a dictionary of keyword arguments to update
                  for the next attempt, or `None` if no changes are needed.

    Returns:
        A decorator that wraps a function with the specified retry logic.

    Raises:
        The original exception if the function fails on its final attempt.
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
            return None

        return wrapper

    return decorator
