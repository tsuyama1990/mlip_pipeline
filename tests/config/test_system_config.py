from pathlib import Path

from mlip_autopipec.config.schemas.common import Composition, MinimalConfig, Resources, TargetSystem
from mlip_autopipec.config.schemas.system import SystemConfig


def test_system_config_auto_population():
    minimal = MinimalConfig(
        project_name="TestProj",
        target_system=TargetSystem(elements=["Fe"], composition=Composition({"Fe": 1.0})),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=4),
    )

    config = SystemConfig(
        minimal=minimal,
        working_dir=Path("/tmp"),
        db_path=Path("/tmp/db.sqlite"),
        log_path=Path("/tmp/log.txt"),
    )

    assert config.target_system is not None
    assert config.target_system.elements == ["Fe"]
    assert config.project_name == "TestProj"
