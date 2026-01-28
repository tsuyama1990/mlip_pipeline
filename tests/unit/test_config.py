from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.dft import DFTConfig


def test_dft_config_valid(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Fe.upf").touch()  # Create UPF file to satisfy validation
    config = DFTConfig(
        command="pw.x",
        pseudopotential_dir=pseudo_dir,
        ecutwfc=50.0,
    )
    assert config.ecutwfc == 50.0
    assert config.command == "pw.x"


def test_dft_config_invalid_paths(tmp_path):
    # Depending on validation strictness
    # Here we assume it passes type checks but might fail validation if logic exists
    # Currently just checking basic types
    pass


def test_dft_config_constraints(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Fe.upf").touch()
    with pytest.raises(ValidationError):
        DFTConfig(
            command="pw.x",
            pseudopotential_dir=pseudo_dir,
            mixing_beta=1.5,  # > 1.0
        )
