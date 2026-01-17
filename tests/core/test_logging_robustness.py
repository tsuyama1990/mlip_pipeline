from pathlib import Path
from unittest.mock import patch

import pytest

from mlip_autopipec.core.logging import setup_logging
from mlip_autopipec.exceptions import LoggingError


def test_setup_logging_permission_error():
    path = Path("/root/protected.log")
    with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
        with pytest.raises(LoggingError) as excinfo:
            setup_logging(path)
        assert "Failed to create log file" in str(excinfo.value)

def test_setup_logging_unexpected_error():
    path = Path("/tmp/test.log")
    with patch("logging.basicConfig", side_effect=Exception("Random error")):
        with pytest.raises(LoggingError) as excinfo:
            setup_logging(path)
        assert "Unexpected error" in str(excinfo.value)
