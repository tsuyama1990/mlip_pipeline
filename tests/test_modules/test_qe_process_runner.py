from pathlib import Path
from unittest.mock import MagicMock, patch

from mlip_autopipec.modules.qe_process_runner import QEProcessRunner


def test_qe_process_runner() -> None:
    """Test the QEProcessRunner."""
    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="", stderr="")
        runner = QEProcessRunner(Path("dummy_dir"))
        process = runner.run()
        mock_subprocess.assert_called_once()
        assert process.returncode == 0
