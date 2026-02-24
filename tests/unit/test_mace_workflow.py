"""Tests for MACE Distillation Workflow."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from ase import Atoms

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.interfaces import (
    DynamicsEngine,
    Oracle,
    StructureGenerator,
    Trainer,
)
from pyacemaker.core.utils import validate_structure_integrity
from pyacemaker.domain_models.models import (
    Potential,
    PotentialType,
    StructureMetadata,
    UncertaintyState,
)
from pyacemaker.modules.mace_workflow import MaceDistillationWorkflow
from pyacemaker.oracle.dataset import DatasetManager
from pyacemaker.oracle.mace_oracle import MaceSurrogateOracle

# Constants
TEST_TARGET_POINTS = 5
TEST_UNCERTAINTY_THRESHOLD = 0.5
TEST_MACE_EPOCHS = 10


def create_dummy_structure(id_val: Any = None, uncertainty: float | None = None) -> StructureMetadata:
    """Create a lightweight dummy structure with validation."""
    s = StructureMetadata(id=id_val or uuid4())
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    s.features["atoms"] = atoms
    if uncertainty is not None:
        s.uncertainty_state = UncertaintyState(gamma_max=uncertainty, gamma_mean=uncertainty)

    validate_structure_integrity(s)
    return s


def streaming_generator_mock(n: int = TEST_TARGET_POINTS) -> Iterator[StructureMetadata]:
    """Generator that yields dummy structures one by one."""
    for _ in range(n):
        yield create_dummy_structure(uncertainty=0.8)  # High uncertainty by default


@pytest.fixture
def base_config(tmp_path: Path) -> PYACEMAKERConfig:
    """Fixture for base configuration."""
    config_dict = {
        "project": {"name": "test", "root_dir": str(tmp_path)},
        "oracle": {
            "dft": {
                "code": "vasp",
                "pseudopotentials": {"Fe": "pot"},
                "command": "run",
            },
            "mace": {"model_path": "medium", "mock": True},
        },
        "distillation": {
            "enable_mace_distillation": True,
            "step1_direct_sampling": {"target_points": TEST_TARGET_POINTS},
            "step2_active_learning": {
                "uncertainty_threshold": TEST_UNCERTAINTY_THRESHOLD,
                "cycles": 1,
                "n_select": 2,
            },
            "step3_mace_finetune": {"epochs": TEST_MACE_EPOCHS},
            "step4_surrogate_sampling": {"target_points": TEST_TARGET_POINTS},
            "step7_pacemaker_finetune": {"enable": True, "weight_dft": 10.0},
        },
        "version": "0.1.0",
    }
    return PYACEMAKERConfig(**config_dict)


def test_mace_distillation_workflow_success(base_config: PYACEMAKERConfig) -> None:
    """Test the full happy path of the 7-step workflow with dual oracles."""
    # Mocks
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_sg.generate_direct_samples.return_value = streaming_generator_mock(TEST_TARGET_POINTS)

    # DFT Oracle (Primary)
    mock_dft_oracle = MagicMock(spec=Oracle)
    mock_dft_oracle.compute_batch.side_effect = lambda structures: (
        s for s in structures
    )  # Identity pass-through

    # MACE Oracle (Surrogate)
    mock_mace_oracle = MagicMock(spec=MaceSurrogateOracle)
    # Mock compute_uncertainty
    mock_mace_oracle.compute_uncertainty.side_effect = lambda structures: (
        create_dummy_structure(uncertainty=0.9) for _ in structures
    )
    # Mock compute_batch (for Step 5)
    mock_mace_oracle.compute_batch.side_effect = lambda structures: (
        s for s in structures
    )

    # Trainers
    mock_trainer = MagicMock(spec=Trainer)
    mock_trainer.train.return_value = Potential(
        path=Path("pot.yace"), type=PotentialType.PACE, version="1.0", metrics={}, parameters={}
    )

    mock_mace_trainer = MagicMock(spec=Trainer)
    mock_mace_trainer.train.return_value = Potential(
        path=Path("mace.model"), type=PotentialType.MACE, version="1.0", metrics={}, parameters={}
    )

    # Dynamics
    mock_dyn = MagicMock(spec=DynamicsEngine)
    mock_dyn.run_exploration.return_value = streaming_generator_mock(TEST_TARGET_POINTS)

    dataset_manager = DatasetManager()
    dataset_path = base_config.project.root_dir / "data" / "dataset.pckl.gzip"

    workflow = MaceDistillationWorkflow(
        config=base_config,
        dataset_manager=dataset_manager,
        dataset_path=dataset_path,
        oracle=mock_dft_oracle,
        mace_oracle=mock_mace_oracle,
        trainer=mock_trainer,
        mace_trainer=mock_mace_trainer,
        dynamics_engine=mock_dyn,
        structure_generator=mock_sg,
        validation_path=Path("val"),
        training_path=Path("train"),
    )

    result = workflow.run()

    assert result.status == "success"

    # Verify Step 2 (AL)
    # DFT Oracle should be used for labeling selected structures
    assert mock_dft_oracle.compute_batch.called, "DFT Oracle should be called for Ground Truth"
    # MACE Oracle should be used for uncertainty
    assert mock_mace_oracle.compute_uncertainty.called, "MACE Oracle should be called for Uncertainty"
    # MACE Trainer should be called
    assert mock_mace_trainer.train.called, "MACE Trainer should be called for Fine-tuning"

    # Verify Step 5 (Surrogate Labeling)
    # MACE Oracle should be used for pseudo-labeling
    assert mock_mace_oracle.compute_batch.called, "MACE Oracle should be called for Surrogate Labeling"

    # Verify Step 6 & 7 (Pacemaker Training)
    # Step 6: Base Training
    # Step 7: Delta Learning
    assert mock_trainer.train.call_count >= 2, "Pacemaker Trainer should be called twice (Base + Delta)"

    # Check Delta Learning call arguments
    args, kwargs = mock_trainer.train.call_args_list[-1]
    assert kwargs.get("weight_dft") == 10.0
    assert kwargs.get("initial_potential") is not None


def test_mace_workflow_oracle_failure(base_config: PYACEMAKERConfig) -> None:
    """Test handling of Oracle failure."""
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_sg.generate_direct_samples.return_value = streaming_generator_mock(TEST_TARGET_POINTS)

    mock_dft_oracle = MagicMock(spec=Oracle)
    mock_mace_oracle = MagicMock(spec=MaceSurrogateOracle)

    # Simulate failure in uncertainty computation
    mock_mace_oracle.compute_uncertainty.side_effect = RuntimeError("Oracle Failed")

    workflow = MaceDistillationWorkflow(
        config=base_config,
        dataset_manager=DatasetManager(),
        dataset_path=Path("ds"),
        oracle=mock_dft_oracle,
        mace_oracle=mock_mace_oracle,
        trainer=MagicMock(),
        mace_trainer=MagicMock(),
        dynamics_engine=MagicMock(),
        structure_generator=mock_sg,
        validation_path=Path("val"),
        training_path=Path("train"),
    )

    result = workflow.run()
    # It should not fail completely, but try to continue or finish AL loop early
    # Step 2 catches Exception and continues loop.
    # If loop finishes (max_cycles), it proceeds to next steps.
    # So run() should return success but AL might not have selected anything.
    assert result.status == "success"
    # Check logs or metrics if possible, but assert success handles the "no crash" requirement.


def test_mace_workflow_recovery_after_failure(base_config: PYACEMAKERConfig) -> None:
    """Test that workflow continues if one AL iteration fails."""
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_sg.generate_direct_samples.return_value = streaming_generator_mock(TEST_TARGET_POINTS)

    mock_dft_oracle = MagicMock(spec=Oracle)
    mock_mace_oracle = MagicMock(spec=MaceSurrogateOracle)

    # Fail first call, succeed second call
    def side_effect(structures):
        if mock_mace_oracle.compute_uncertainty.call_count == 1:
            raise RuntimeError("Temporary Failure")
        yield from (create_dummy_structure(uncertainty=0.9) for _ in structures)

    mock_mace_oracle.compute_uncertainty.side_effect = side_effect
    # Mock compute_batch
    mock_mace_oracle.compute_batch.side_effect = lambda s: s

    # Ensure multiple cycles
    base_config.distillation.step2_active_learning.cycles = 2

    workflow = MaceDistillationWorkflow(
        config=base_config,
        dataset_manager=DatasetManager(),
        dataset_path=Path("ds"),
        oracle=mock_dft_oracle,
        mace_oracle=mock_mace_oracle,
        trainer=MagicMock(),
        mace_trainer=MagicMock(),
        dynamics_engine=MagicMock(),
        structure_generator=mock_sg,
        validation_path=Path("val"),
        training_path=Path("train"),
    )

    result = workflow.run()
    assert result.status == "success"
    # Should have tried twice
    assert mock_mace_oracle.compute_uncertainty.call_count >= 2
