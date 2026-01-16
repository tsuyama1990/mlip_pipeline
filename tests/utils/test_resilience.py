"""
Unit tests for the resilience utilities, such as the @retry decorator.
"""
import time
from unittest.mock import Mock, patch

import pytest

from mlip_autopipec.utils.resilience import retry


class CustomError(Exception):
    """A custom exception for testing purposes."""


def test_retry_succeeds_on_first_try():
    """Tests that the decorator calls the function once if it succeeds."""
    mock_func = Mock(return_value="success")
    mock_func.__name__ = "mock_func"
    decorated_func = retry(attempts=3, delay=0.1, exceptions=(CustomError,))(mock_func)
    result = decorated_func()
    assert result == "success"
    mock_func.assert_called_once()


def test_retry_succeeds_after_failures():
    """Tests that the decorator retries on failure and succeeds eventually."""
    mock_func = Mock(side_effect=[CustomError, CustomError, "success"])
    mock_func.__name__ = "mock_func"
    decorated_func = retry(attempts=3, delay=0.1, exceptions=(CustomError,))(mock_func)
    result = decorated_func()
    assert result == "success"
    assert mock_func.call_count == 3


def test_retry_fails_after_max_attempts():
    """Tests that the decorator gives up and re-raises the exception."""
    mock_func = Mock(side_effect=CustomError)
    mock_func.__name__ = "mock_func"
    decorated_func = retry(attempts=3, delay=0.1, exceptions=(CustomError,))(mock_func)
    with pytest.raises(CustomError):
        decorated_func()
    assert mock_func.call_count == 3


def test_retry_ignores_different_exception():
    """Tests that an exception not in the list is not caught."""
    mock_func = Mock(side_effect=ValueError("Different error"))
    mock_func.__name__ = "mock_func"
    decorated_func = retry(attempts=3, delay=0.1, exceptions=(CustomError,))(mock_func)
    with pytest.raises(ValueError, match="Different error"):
        decorated_func()
    mock_func.assert_called_once()


@patch("time.sleep")
def test_retry_waits_for_delay(mock_sleep):
    """Tests that the decorator waits for the specified delay between retries."""
    mock_func = Mock(side_effect=[CustomError, "success"])
    mock_func.__name__ = "mock_func"
    decorated_func = retry(attempts=2, delay=0.5, exceptions=(CustomError,))(mock_func)
    decorated_func()
    mock_sleep.assert_called_once_with(0.5)
    assert mock_func.call_count == 2


def test_on_retry_callback_is_called_and_modifies_kwargs(monkeypatch):
    """
    Tests that the on_retry callback is correctly called and that its return
    value is used to update the arguments for the next attempt.
    """
    monkeypatch.setattr(time, "sleep", lambda x: None)  # Patch time.sleep

    mock_func = Mock(side_effect=[CustomError, "success"])
    mock_func.__name__ = "mock_func"

    def on_retry_handler(exception, kwargs):
        assert isinstance(exception, CustomError)
        assert kwargs["param"] == "initial"
        return {"param": "modified"}

    decorated_func = retry(
        attempts=2, delay=0.1, exceptions=(CustomError,), on_retry=on_retry_handler
    )(mock_func)

    decorated_func(param="initial")

    # Check that the function was called twice
    assert mock_func.call_count == 2
    # Check that the first call had the initial parameter
    mock_func.assert_any_call(param="initial")
    # Check that the second call had the modified parameter
    mock_func.assert_any_call(param="modified")
