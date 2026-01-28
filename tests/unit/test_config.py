from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.core import TargetSystem
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.config.schemas.training import TrainingConfig


def test_target_system_valid():
    ts = TargetSystem(name="TestSys", elements=["Si"], lattices=["fcc"], config={"key": "value"})
    assert ts.name == "TestSys"


def test_target_system_invalid_elements():
    with pytest.raises(ValidationError):
        TargetSystem(
            name="BadSys",
            elements=["Xy"],  # Invalid element
            lattices=["fcc"],
        )


def test_dft_config_security():
    # Valid
    p_dir = Path("/tmp")
    config = DFTConfig(pseudopotential_dir=p_dir, command="pw.x")
    assert config.command == "pw.x"

    # Invalid injection
    with pytest.raises(ValidationError):
        DFTConfig(pseudopotential_dir=p_dir, command="pw.x; rm -rf /")


def test_dft_config_defaults(tmp_path):
    p_dir = tmp_path / "pseudos"
    p_dir.mkdir()
    (p_dir / "Si.upf").touch()

    config = DFTConfig(pseudopotential_dir=p_dir)
    assert config.ecutwfc == 60.0
    assert config.command == "pw.x"


def test_training_config_defaults():
    config = TrainingConfig(potential_name="test_pot", cutoff=5.0)
    assert config.batch_size == 32
