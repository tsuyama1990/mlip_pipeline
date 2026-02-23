"""Tests for MACE Distillation Workflow."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from ase import Atoms
from pydantic import ValidationError

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import MaceConfig, PYACEMAKERConfig
from pyacemaker.core.interfaces import (
    DynamicsEngine,
    Oracle,
    StructureGenerator,
    Trainer,
    UncertaintyModel,
    Validator,
)
from pyacemaker.core.utils import validate_structure_integrity
from pyacemaker.domain_models.models import (
    Potential,
    PotentialType,
    StructureMetadata,
    StructureStatus,
    UncertaintyState,
)
from pyacemaker.oracle.mace_manager import MaceManager
from pyacemaker.orchestrator import Orchestrator

# Constants
TEST_TARGET_POINTS = 5
TEST_UNCERTAINTY_THRESHOLD = 0.5
TEST_MACE_EPOCHS = 10

def create_dummy_structure(id_val: Any = None, uncertainty: float | None = None) -> StructureMetadata:
    """Create a lightweight dummy structure with validation."""
    s = StructureMetadata(id=id_val or uuid4())
    atoms = Atoms("Fe", positions=[[0,0,0]], cell=[2,2,2], pbc=True)
    s.features["atoms"] = atoms
    if uncertainty is not None:
        s.uncertainty_state = UncertaintyState(gamma_max=uncertainty, gamma_mean=uncertainty)

    validate_structure_integrity(s)
    return s

def streaming_generator_mock(n: int = TEST_TARGET_POINTS) -> Iterator[StructureMetadata]:
    """Generator that yields dummy structures one by one."""
    for _ in range(n):
        yield create_dummy_structure()

class MockOracle(Oracle, UncertaintyModel):
    """Mock Oracle implementing both interfaces."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        super().__init__(config)
        self.fail_uncertainty = False

    def run(self) -> ModuleResult:
        return ModuleResult(status="success", metrics=Metrics())

    def compute_batch(self, structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
        for s in structures:
            s.status = StructureStatus.CALCULATED
            s.energy = -1.0
            if "atoms" in s.features:
                n_atoms = len(s.features["atoms"])
                s.forces = [[0.0, 0.0, 0.0] for _ in range(n_atoms)]
            yield s

    def compute_uncertainty(self, structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
        if self.fail_uncertainty:
            msg = "Oracle Failed"
            raise RuntimeError(msg)
        for s in structures:
            s.uncertainty_state = UncertaintyState(gamma_max=0.9, gamma_mean=0.5)
            yield s

@pytest.fixture
def base_config(tmp_path: Path) -> PYACEMAKERConfig:
    """Fixture for base configuration."""
    config_dict = {
        "project": {"name": "test", "root_dir": str(tmp_path)},
        "oracle": {
            "dft": {
                "code": "vasp",
                "pseudopotentials": {"Fe": "pot"},
                "command": "run"
            },
            "mace": {"model_path": "medium", "mock": True}
        },
        "distillation": {
            "enable_mace_distillation": True,
            "step1_direct_sampling": {"target_points": TEST_TARGET_POINTS},
            "step2_active_learning": {
                "uncertainty_threshold": TEST_UNCERTAINTY_THRESHOLD,
                "cycles": 1,
                "n_select": 2
            },
            "step3_mace_finetune": {"epochs": TEST_MACE_EPOCHS},
            "step4_surrogate_sampling": {"target_points": TEST_TARGET_POINTS}
        },
        "version": "0.1.0"
    }
    return PYACEMAKERConfig(**config_dict)

def test_mace_distillation_workflow_success(base_config: PYACEMAKERConfig) -> None:
    """Test the full happy path of the 7-step workflow."""
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_sg.generate_direct_samples.side_effect = lambda n_samples, objective: streaming_generator_mock(n_samples)

    mock_oracle = MockOracle(base_config)

    mock_trainer = MagicMock(spec=Trainer)
    mock_trainer.train.return_value = Potential(path=Path("pot.yace"), type=PotentialType.PACE, version="1.0", metrics={}, parameters={})

    mock_dyn = MagicMock(spec=DynamicsEngine)
    mock_dyn.run_exploration.side_effect = lambda pot, seeds: streaming_generator_mock(TEST_TARGET_POINTS)

    mock_val = MagicMock(spec=Validator)

    mock_mace_trainer = MagicMock(spec=Trainer)
    mock_mace_trainer.train.return_value = Potential(path=Path("mace.model"), type=PotentialType.MACE, version="1.0", metrics={}, parameters={})

    orch = Orchestrator(
        base_config,
        structure_generator=mock_sg,
        oracle=mock_oracle,
        trainer=mock_trainer,
        dynamics_engine=mock_dyn,
        validator=mock_val,
        mace_trainer=mock_mace_trainer
    )

    orch.uncertainty_model = mock_oracle

    from pyacemaker.workflows.distillation import MaceDistillationWorkflow

    orch.distillation_workflow = MaceDistillationWorkflow(
        base_config,
        orch.dataset_manager,
        mock_sg,
        mock_oracle,
        mock_mace_trainer,
        orch.active_learner,
        mock_oracle,
        mock_dyn,
        mock_trainer
    )

    result = orch.run()

    assert result.status == "success"

    mock_sg.generate_direct_samples.assert_called_once()
    assert mock_dyn.run_exploration.called
    assert mock_mace_trainer.train.called
    assert mock_trainer.train.call_count >= 1

def test_mace_workflow_early_convergence(base_config: PYACEMAKERConfig) -> None:
    """Test that Step 2 loop breaks early if uncertainty is low."""
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_sg.generate_direct_samples.side_effect = lambda **kwargs: streaming_generator_mock(5)

    mock_oracle = MockOracle(base_config)

    def low_uncertainty(structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
        for s in structures:
            s.uncertainty_state = UncertaintyState(gamma_max=0.1, gamma_mean=0.1)
            yield s

    mock_oracle.compute_uncertainty = low_uncertainty # type: ignore

    mock_mace_trainer = MagicMock(spec=Trainer)

    orch = Orchestrator(
        base_config,
        structure_generator=mock_sg,
        oracle=mock_oracle,
        trainer=MagicMock(),
        dynamics_engine=MagicMock(),
        mace_trainer=mock_mace_trainer
    )

    from pyacemaker.workflows.distillation import MaceDistillationWorkflow
    orch.distillation_workflow = MaceDistillationWorkflow(
        base_config,
        orch.dataset_manager,
        mock_sg,
        mock_oracle,
        mock_mace_trainer,
        orch.active_learner,
        mock_oracle,
        MagicMock(),
        MagicMock()
    )

    orch.run()

    mock_mace_trainer.train.assert_not_called()

def test_mace_workflow_oracle_failure(base_config: PYACEMAKERConfig) -> None:
    """Test handling of Oracle failure."""
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_sg.generate_direct_samples.side_effect = lambda **kwargs: streaming_generator_mock(5)

    mock_oracle = MockOracle(base_config)
    mock_oracle.fail_uncertainty = True

    orch = Orchestrator(
        base_config,
        structure_generator=mock_sg,
        oracle=mock_oracle,
        trainer=MagicMock(),
        dynamics_engine=MagicMock(),
        mace_trainer=MagicMock()
    )

    from pyacemaker.workflows.distillation import MaceDistillationWorkflow
    orch.distillation_workflow = MaceDistillationWorkflow(
        base_config,
        orch.dataset_manager,
        mock_sg,
        mock_oracle,
        MagicMock(),
        orch.active_learner,
        mock_oracle,
        MagicMock(),
        MagicMock()
    )

    with pytest.raises(RuntimeError, match="Oracle Failed"):
        orch.run()

def test_config_validation_mace_mode(base_config: PYACEMAKERConfig) -> None:
    """Test that invalid configuration prevents running."""
    class PlainOracle(Oracle):
        def __init__(self, c: PYACEMAKERConfig) -> None: super().__init__(c)
        def run(self) -> ModuleResult: return ModuleResult(status="ok", metrics=Metrics())
        def compute_batch(self, s: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]: yield from s

    plain_oracle = PlainOracle(base_config)
    base_config.oracle.mace = None

    with pytest.raises(RuntimeError, match="Missing components"):
        Orchestrator(base_config, oracle=plain_oracle)

def test_empty_generator_handling(base_config: PYACEMAKERConfig) -> None:
    """Test handling of empty generator from structure generator."""
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_sg.generate_direct_samples.return_value = iter([])

    mock_oracle = MockOracle(base_config)

    mock_trainer = MagicMock(spec=Trainer)
    mock_trainer.train.return_value = Potential(path=Path("mock.pot"), type=PotentialType.PACE, version="1.0", metrics={}, parameters={})

    mock_dyn = MagicMock(spec=DynamicsEngine)
    mock_dyn.run_exploration.return_value = iter([])

    orch = Orchestrator(
        base_config,
        structure_generator=mock_sg,
        oracle=mock_oracle,
        trainer=mock_trainer,
        dynamics_engine=mock_dyn,
        mace_trainer=MagicMock()
    )

    from pyacemaker.workflows.distillation import MaceDistillationWorkflow
    orch.distillation_workflow = MaceDistillationWorkflow(
        base_config,
        orch.dataset_manager,
        mock_sg,
        mock_oracle,
        MagicMock(),
        orch.active_learner,
        mock_oracle,
        mock_dyn,
        mock_trainer
    )

    res = orch.run()
    assert res.status == "success"

def test_mace_workflow_malformed_uncertainty(base_config: PYACEMAKERConfig) -> None:
    """Test handling of structures with missing/malformed uncertainty state."""
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_sg.generate_direct_samples.side_effect = lambda **kwargs: streaming_generator_mock(5)

    mock_oracle = MockOracle(base_config)
    mock_oracle.compute_batch = MagicMock(wraps=mock_oracle.compute_batch) # type: ignore

    def malformed_uncertainty(structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
        for s in structures:
            s.uncertainty_state = None
            yield s

    mock_oracle.compute_uncertainty = malformed_uncertainty # type: ignore

    mock_trainer = MagicMock(spec=Trainer)
    mock_trainer.train.return_value = Potential(path=Path("pot.yace"), type=PotentialType.PACE, version="1.0", metrics={}, parameters={})

    mock_dyn = MagicMock(spec=DynamicsEngine)
    mock_dyn.run_exploration.return_value = iter([])

    orch = Orchestrator(
        base_config,
        structure_generator=mock_sg,
        oracle=mock_oracle,
        trainer=mock_trainer,
        dynamics_engine=mock_dyn,
        mace_trainer=MagicMock()
    )

    from pyacemaker.workflows.distillation import MaceDistillationWorkflow
    orch.distillation_workflow = MaceDistillationWorkflow(
        base_config,
        orch.dataset_manager,
        mock_sg,
        mock_oracle,
        MagicMock(),
        orch.active_learner,
        mock_oracle,
        mock_dyn,
        mock_trainer
    )

    result = orch.run()

    assert result.status == "success"
    mock_oracle.compute_batch.assert_not_called()

def test_mace_security_validation() -> None:
    """Test MACE manager security validation."""
    config = MaceConfig(model_path="medium")
    _ = MaceManager(config)

    with pytest.raises(ValidationError, match="Invalid model path structure"):
        MaceConfig(model_path="/../../etc/passwd")
