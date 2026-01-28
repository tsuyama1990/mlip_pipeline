import pytest
from pydantic import ValidationError

from mlip_autopipec.config.models import MLIPConfig
from mlip_autopipec.config.schemas.core import TargetSystem
from mlip_autopipec.config.schemas.dft import DFTConfig


def test_target_system_valid():
    ts = TargetSystem(
        name="TestSys",
        elements=["Fe", "Ni"],
        composition={"Fe": 0.5, "Ni": 0.5},
        crystal_structure="fcc",
    )
    assert ts.name == "TestSys"
    assert ts.elements == ["Fe", "Ni"]


def test_target_system_invalid_composition_sum():
    with pytest.raises(ValidationError):
        TargetSystem(elements=["Fe"], composition={"Fe": 0.9})


def test_target_system_invalid_element():
    with pytest.raises(ValidationError):
        TargetSystem(elements=["Xy"], composition={"Xy": 1.0})


def test_dft_config_defaults(tmp_path):
    (tmp_path / "fake.upf").touch()
    dft = DFTConfig(pseudopotential_dir=tmp_path, ecutwfc=30.0, kspacing=0.1)
    assert dft.command == "pw.x"
    assert dft.nspin == 1


def test_dft_config_validation(tmp_path):
    (tmp_path / "fake.upf").touch()
    with pytest.raises(ValidationError) as exc:
        DFTConfig(
            pseudopotential_dir=tmp_path,
            ecutwfc=-10.0,  # Invalid
            kspacing=0.1,
        )
    assert "greater than 0" in str(exc.value)


def test_dft_config_security(tmp_path):
    (tmp_path / "fake.upf").touch()
    with pytest.raises(ValidationError) as exc:
        DFTConfig(
            pseudopotential_dir=tmp_path, ecutwfc=30.0, kspacing=0.1, command="pw.x; rm -rf /"
        )
    assert "unsafe shell characters" in str(exc.value)


def test_mlip_config_full(tmp_path):
    (tmp_path / "Al.upf").touch()
    config_data = {
        "target_system": {
            "name": "Al System", # REQUIRED field was missing in test_config.py causing validation error? No, name is optional/default? Wait, checking TargetSystem.
            # Looking at schemas/core.py: name: str = Field(..., description="System name") -> REQUIRED.
            # So I must provide it.
            "elements": ["Al"],
            "composition": {"Al": 1.0},
            "crystal_structure": "fcc",
        },
        "dft": {"pseudopotential_dir": str(tmp_path), "ecutwfc": 40.0, "kspacing": 0.1},
        "runtime": {"database_path": "test.db", "work_dir": "work"},
    }
    config = MLIPConfig(**config_data)
    assert config.target_system.elements == ["Al"]
    assert config.runtime.database_path.name == "test.db"
