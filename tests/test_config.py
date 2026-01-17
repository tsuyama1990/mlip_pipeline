from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.models import MinimalConfig, Resources, SystemConfig, TargetSystem


def test_target_system_valid() -> None:
    ts = TargetSystem(elements=["Fe", "Ni"], composition={"Fe": 0.6, "Ni": 0.4})
    assert ts.elements == ["Fe", "Ni"]
    assert ts.composition["Fe"] == 0.6


def test_target_system_invalid_element() -> None:
    with pytest.raises(ValidationError) as exc:
        TargetSystem(elements=["Fe", "Xy"], composition={"Fe": 0.6, "Xy": 0.4})
    assert "Invalid chemical symbol: Xy" in str(exc.value)


def test_target_system_invalid_composition_sum() -> None:
    with pytest.raises(ValidationError) as exc:
        TargetSystem(elements=["Fe", "Ni"], composition={"Fe": 0.5, "Ni": 0.4})
    assert "Composition must sum to 1.0" in str(exc.value)


def test_resources_valid() -> None:
    res = Resources(dft_code="quantum_espresso", parallel_cores=4, gpu_enabled=True)
    assert res.dft_code == "quantum_espresso"
    assert res.parallel_cores == 4
    assert res.gpu_enabled is True


def test_resources_invalid_cores() -> None:
    with pytest.raises(ValidationError):
        Resources(dft_code="vasp", parallel_cores=0)


def test_minimal_config_valid() -> None:
    conf = MinimalConfig(
        project_name="TestProject",
        target_system=TargetSystem(elements=["Al"], composition={"Al": 1.0}),
        resources=Resources(dft_code="vasp", parallel_cores=1),
    )
    assert conf.project_name == "TestProject"


def test_system_config_immutability(tmp_path: Path) -> None:
    minimal = MinimalConfig(
        project_name="Test",
        target_system=TargetSystem(elements=["Cu"], composition={"Cu": 1.0}),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=2),
    )
    sys_conf = SystemConfig(
        minimal=minimal,
        working_dir=tmp_path,
        db_path=tmp_path / "db.sqlite",
        log_path=tmp_path / "log.txt",
    )

    with pytest.raises(ValidationError):
        sys_conf.working_dir = Path("/new/path")
