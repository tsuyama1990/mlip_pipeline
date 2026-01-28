from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.dft.runner import DFTFatalError, QERunner


@pytest.fixture
def mock_config(tmp_path: Path) -> DFTConfig:
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Al.UPF").touch()
    return DFTConfig(
        pseudopotential_dir=pseudo_dir,
        ecutwfc=30.0,
        kspacing=0.05,
        command="pw.x",
        recoverable=True,
    )


@patch("shutil.which")
def test_validate_command_forbidden_chars(mock_which: MagicMock, mock_config: DFTConfig) -> None:
    mock_which.return_value = "/bin/pw.x"
    runner = QERunner(mock_config)

    # Test various injection attempts
    injections = [
        "pw.x; rm -rf /",
        "pw.x && echo 'hack'",
        "pw.x | bash",
        "pw.x `whoami`",
        "pw.x $(ls)",
        "pw.x > output",
        "pw.x < input",
    ]

    for cmd in injections:
        with pytest.raises(DFTFatalError, match="unsafe shell characters"):
            runner._validate_command(cmd)


@patch("shutil.which")
def test_validate_command_valid_args(mock_which: MagicMock, mock_config: DFTConfig) -> None:
    mock_which.return_value = "/bin/pw.x"
    runner = QERunner(mock_config)

    # Valid arguments should pass
    parts = runner._validate_command("pw.x -np 4 -in pw.in")
    assert parts == ["pw.x", "-np", "4", "-in", "pw.in"]
