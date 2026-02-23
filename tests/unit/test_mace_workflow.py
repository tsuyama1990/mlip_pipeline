"""Tests for MACE Distillation Workflow."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from ase import Atoms

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.interfaces import (
    DynamicsEngine,
    Oracle,
    StructureGenerator,
    Trainer,
    UncertaintyModel,
)
from pyacemaker.core.utils import validate_structure_integrity
from pyacemaker.domain_models.models import (
    Potential,
    PotentialType,
    StructureMetadata,
    StructureStatus,
    UncertaintyState,
)
from pyacemaker.modules.mace_workflow import MaceDistillationWorkflow
from pyacemaker.oracle.dataset import DatasetManager

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

    mock_mace_trainer = MagicMock(spec=Trainer)
    mock_mace_trainer.train.return_value = Potential(path=Path("mace.model"), type=PotentialType.MACE, version="1.0", metrics={}, parameters={})

    dataset_manager = DatasetManager()
    dataset_path = base_config.project.root_dir / "data" / "dataset.pckl.gzip"

    workflow = MaceDistillationWorkflow(
        config=base_config,
        dataset_manager=dataset_manager,
        dataset_path=dataset_path,
        oracle=mock_oracle,
        trainer=mock_trainer,
        mace_trainer=mock_mace_trainer,
        dynamics_engine=mock_dyn,
        structure_generator=mock_sg,
        validation_path=Path("val"),
        training_path=Path("train")
    )

    with patch('pyacemaker.modules.mace_workflow.MaceSurrogateOracle') as MockMaceOracleCls, \
         patch('pyacemaker.modules.mace_workflow.DirectGenerator') as MockDirectGen:

        mock_direct_instance = MockDirectGen.return_value
        mock_direct_instance.generate_direct_samples.side_effect = mock_sg.generate_direct_samples

        mock_mace_instance = MockMaceOracleCls.return_value
        def mock_compute_batch(structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
            for s in structures:
                s.status = StructureStatus.CALCULATED
                s.energy = -2.0
                n_atoms = len(s.features.get("atoms", []))
                s.forces = [[0.1, 0.1, 0.1] for _ in range(n_atoms)]
                yield s
        mock_mace_instance.compute_batch.side_effect = mock_compute_batch

        result = workflow.run()

    assert result.status == "success"
    assert mock_dyn.run_exploration.called
    assert mock_mace_trainer.train.called
    assert mock_trainer.train.call_count >= 1

def test_mace_workflow_early_convergence(base_config: PYACEMAKERConfig) -> None:
    """Test that Step 2 loop breaks early if uncertainty is low."""
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_sg.generate_direct_samples.side_effect = lambda **kwargs: streaming_generator_mock(5)

    mock_oracle = MockOracle(base_config)
    # Return low uncertainty
    def low_uncertainty(structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
        for s in structures:
            s.uncertainty_state = UncertaintyState(gamma_max=0.1, gamma_mean=0.1)
            yield s
    mock_oracle.compute_uncertainty = low_uncertainty # type: ignore

    mock_mace_trainer = MagicMock(spec=Trainer)

    workflow = MaceDistillationWorkflow(
        config=base_config,
        dataset_manager=DatasetManager(),
        dataset_path=base_config.project.root_dir / "data" / "ds.pckl",
        oracle=mock_oracle,
        trainer=MagicMock(),
        mace_trainer=mock_mace_trainer,
        dynamics_engine=MagicMock(),
        structure_generator=mock_sg,
        validation_path=Path("val"),
        training_path=Path("train")
    )

    with patch('pyacemaker.modules.mace_workflow.MaceSurrogateOracle'), \
         patch('pyacemaker.modules.mace_workflow.DirectGenerator') as MockDirectGen:

        MockDirectGen.return_value.generate_direct_samples.side_effect = mock_sg.generate_direct_samples
        workflow.run() # Testing run() which calls _step2...

    # MACE trainer should NOT have trained in Step 2 (AL loop) because it converged early
    # But wait, step 2 loop calls `_finetune_mace` if `selected` is True.
    # Here `selected` will be empty because uncertainty < threshold (0.5).
    # So `_finetune_mace` is NOT called inside the loop.
    # However, Step 3 is "integrated" into Step 2 loop in this architecture.
    # The loop breaks.

    # But later steps?
    # Step 6 and 7 call trainer.train.
    # Does Step 3 (MACE finetune) happen outside the loop? No, it's inside `_step2_active_learning_loop`.
    # So mock_mace_trainer.train should NOT be called.
    mock_mace_trainer.train.assert_not_called()

def test_mace_workflow_oracle_failure(base_config: PYACEMAKERConfig) -> None:
    """Test handling of Oracle failure."""
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_sg.generate_direct_samples.side_effect = lambda **kwargs: streaming_generator_mock(5)

    mock_oracle = MockOracle(base_config)
    mock_oracle.fail_uncertainty = True

    workflow = MaceDistillationWorkflow(
        config=base_config,
        dataset_manager=DatasetManager(),
        dataset_path=Path("ds"),
        oracle=mock_oracle,
        trainer=MagicMock(),
        mace_trainer=MagicMock(),
        dynamics_engine=MagicMock(),
        structure_generator=mock_sg,
        validation_path=Path("val"),
        training_path=Path("train")
    )

    # We need to patch DirectGenerator because run() calls _step1 which uses it
    with (
        patch('pyacemaker.modules.mace_workflow.DirectGenerator'),
        pytest.raises(RuntimeError, match="Oracle Failed"),
    ):
        workflow.run()
