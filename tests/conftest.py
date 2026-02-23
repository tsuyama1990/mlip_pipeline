"""Global pytest configuration."""

from pathlib import Path

import pytest

from pyacemaker.core.config import (
    CONSTANTS,
    DFTConfig,
    DynamicsEngineConfig,
    MaceConfig,
    OracleConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    TrainerConfig,
)


@pytest.fixture(autouse=True)
def skip_file_security_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip file security checks for tests running in /tmp."""
    # By default, tests use tmp_path which is outside CWD (/app).
    # We must allow this for tests to pass.
    # Security tests should explicitly override this or use valid paths.
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", True)


@pytest.fixture
def full_config(tmp_path: Path) -> PYACEMAKERConfig:
    """Return a full configuration object."""
    return PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="TestProject", root_dir=tmp_path),
        oracle=OracleConfig(
            dft=DFTConfig(
                code="mock_dft",
                pseudopotentials={"Fe": "Fe.pbe.UPF"},
            ),
            mace=MaceConfig(model_path="medium", device="cpu"),
        ),
        trainer=TrainerConfig(
            potential_type="mace",
            mock=True,
        ),
        dynamics_engine=DynamicsEngineConfig(
            engine="ase",
            gamma_threshold=0.5,
        ),
    )
