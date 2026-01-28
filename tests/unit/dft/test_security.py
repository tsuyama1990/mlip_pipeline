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
        # Providing explicit values to satisfy strict type checks
        nspin=1,
        diagonalization="david",
        smearing="mv",
        degauss=0.02,
        max_retries=5,
        timeout=3600,
        pseudopotentials=None,
    )


@patch("shutil.which")
def test_validate_command_forbidden_chars(mock_which: MagicMock, mock_config: DFTConfig, tmp_path: Path) -> None:
    mock_which.return_value = "/bin/pw.x"
    # QERunner init requires work_dir
    runner = QERunner(mock_config, work_dir=tmp_path)

    # Test various injection attempts
    injections = [
        "pw.x; rm -rf /",
        "pw.x && echo 'hack'",
        "pw.x | bash",
        "pw.x `whoami`",
        "pw.x $(ls)",
    ]

    for cmd in injections:
        with pytest.raises(DFTFatalError, match="unsafe shell characters"):
            runner._validate_command(cmd)


@patch("shutil.which")
def test_validate_command_valid_args(mock_which: MagicMock, mock_config: DFTConfig, tmp_path: Path) -> None:
    mock_which.return_value = "/bin/pw.x"
    runner = QERunner(mock_config, work_dir=tmp_path)

    # Valid arguments should pass
    parts = runner._validate_command("pw.x -np 4 -in pw.in")
    assert parts == ["pw.x", "-np", "4", "-in", "pw.in"]
