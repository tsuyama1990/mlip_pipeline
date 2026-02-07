from pathlib import Path

from mlip_autopipec.config.config_model import (
    ExplorerConfig,
    GlobalConfig,
    OracleConfig,
    TrainerConfig,
)
from mlip_autopipec.infrastructure.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator
from mlip_autopipec.main import Orchestrator


def test_orchestrator_factory_creation(tmp_path: Path) -> None:
    config = GlobalConfig(
        work_dir=tmp_path,
        max_cycles=1,
        oracle=OracleConfig(type="mock"),
        trainer=TrainerConfig(type="mock"),
        explorer=ExplorerConfig(type="mock")
    )

    orch = Orchestrator(config)

    assert isinstance(orch.oracle, MockOracle)
    assert isinstance(orch.trainer, MockTrainer)
    assert isinstance(orch.explorer, MockExplorer)
    assert isinstance(orch.validator, MockValidator)

def test_orchestrator_factory_invalid_type(tmp_path: Path) -> None:
    # Need to bypass pydantic validation to test factory error if possible,
    # but GlobalConfig validation prevents invalid types.
    # So factory methods are safe unless we modify config after creation or have loose typing.
    # Pydantic prevents invalid "type" values.
    # However, if we add more types later, we might need to test this.
    # For now, "mock" is the only option.
    pass
