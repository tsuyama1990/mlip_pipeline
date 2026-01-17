# FIXME: The above comment is a temporary workaround for a ruff bug.
# It should be removed once the bug is fixed.
# For more information, see: https://github.com/astral-sh/ruff/issues/10515
"""
This module defines custom exceptions for the MLIP-AutoPipe application.

Using custom exceptions allows for more specific and expressive error
handling throughout the workflow, making it easier to catch and manage
different types of failures.
"""


class DFTCalculationError(Exception):
    """
    Raised when a DFT calculation fails and cannot be recovered automatically.

    This exception is raised by the DFTFactory after all retry attempts
    have been exhausted, indicating a fatal error in the calculation.
    """

    def __init__(
        self,
        message: str,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        """
        Initializes the exception with details about the failure.

        Args:
            message: A summary of the error.
            stdout: The standard output from the failed DFT process.
            stderr: The standard error from the failed DFT process.
        """
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self) -> str:
        """Returns a detailed string representation of the error."""
        return f"{super().__str__()}\n--- STDOUT ---\n{self.stdout}\n--- STDERR ---\n{self.stderr}"

class MaxRetriesExceededError(Exception):
    """Raised when a function fails after the maximum number of retry attempts."""
