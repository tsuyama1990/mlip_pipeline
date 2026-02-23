"""Tests for MACE Distillation Workflow."""

import pytest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
from uuid import uuid4
from itertools import repeat

from pyacemaker.core.config import PYACEMAKERConfig, DistillationConfig
from pyacemaker.domain_models.models import StructureMetadata, StructureStatus, UncertaintyState, Potential, CycleStatus
from pyacemaker.orchestrator import Orchestrator
from pyacemaker.core.interfaces import StructureGenerator, Oracle, UncertaintyModel, Trainer, DynamicsEngine, Validator
from pyacemaker.core.base import ModuleResult, Metrics
from ase import Atoms

# Constants for testing
TEST_TARGET_POINTS = 5
TEST_UNCERTAINTY_THRESHOLD = 0.5
TEST_MACE_EPOCHS = 10

def create_dummy_structure(id_val=None, uncertainty=None):
    """Create a lightweight dummy structure."""
    s = StructureMetadata(id=id_val or uuid4())
    s.features["atoms"] = Atoms("Fe", positions=[[0,0,0]], cell=[2,2,2])
    if uncertainty is not None:
        s.uncertainty_state = UncertaintyState(gamma_max=uncertainty, gamma_mean=uncertainty)
    return s

def generator_mock(n=TEST_TARGET_POINTS):
    """Generator that yields dummy structures."""
    for _ in range(n):
        yield create_dummy_structure()

class MockOracle(Oracle, UncertaintyModel):
    """Mock Oracle implementing both interfaces."""
    def __init__(self, config) -> None:
        super().__init__(config)
        self.fail_uncertainty = False

    def compute_batch(self, structures):
        for s in structures:
            s.status = StructureStatus.CALCULATED
            s.energy = -1.0
            s.forces = [[0.0, 0.0, 0.0]]
            yield s

    def compute_uncertainty(self, structures):
        if self.fail_uncertainty:
            raise RuntimeError("Oracle Failed")
        for s in structures:
            # Assign random high uncertainty to ensure selection
            s.uncertainty_state = UncertaintyState(gamma_max=0.9, gamma_mean=0.5)
            yield s

    def run(self):
        return ModuleResult(status="success", metrics=Metrics())

@pytest.fixture
def base_config(tmp_path):
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
            "step2_active_learning": {"uncertainty_threshold": TEST_UNCERTAINTY_THRESHOLD},
            "step3_mace_finetune": {"epochs": TEST_MACE_EPOCHS},
            "step4_surrogate_sampling": {"target_points": TEST_TARGET_POINTS}
        },
        "version": "0.1.0"
    }
    return PYACEMAKERConfig(**config_dict)

def test_mace_distillation_workflow_success(base_config):
    """Test the full happy path of the 7-step workflow."""

    # Mock Modules
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_sg.generate_direct_samples.side_effect = lambda n_samples, objective: generator_mock(n_samples)

    mock_oracle = MockOracle(base_config)

    mock_trainer = MagicMock(spec=Trainer)
    mock_trainer.train.return_value = Potential(path=Path("pot.yace"), type="PACE", version="1.0", metrics={}, parameters={})

    mock_dyn = MagicMock(spec=DynamicsEngine)
    mock_dyn.run_exploration.side_effect = lambda pot, seeds: generator_mock(TEST_TARGET_POINTS)

    mock_val = MagicMock(spec=Validator)

    mock_mace_trainer = MagicMock(spec=Trainer)
    mock_mace_trainer.train.return_value = Potential(path=Path("mace.model"), type="MACE", version="1.0", metrics={}, parameters={})

    orch = Orchestrator(
        base_config,
        structure_generator=mock_sg,
        oracle=mock_oracle,
        trainer=mock_trainer,
        dynamics_engine=mock_dyn,
        validator=mock_val,
        mace_trainer=mock_mace_trainer
    )

    # Patch the internal surrogate oracle instantiation
    with patch('pyacemaker.orchestrator.MaceSurrogateOracle') as MockMaceOracleCls:
        mock_mace_instance = MockMaceOracleCls.return_value
        def mock_compute_batch(structures):
            for s in structures:
                s.status = StructureStatus.CALCULATED
                s.energy = -2.0
                s.forces = [[0.1, 0.1, 0.1]]
                yield s
        mock_mace_instance.compute_batch.side_effect = mock_compute_batch

        result = orch.run()

    assert result.status == "success"

    # Validation
    mock_sg.generate_direct_samples.assert_called_once()
    assert mock_dyn.run_exploration.called
    assert mock_mace_trainer.train.called
    assert mock_trainer.train.call_count >= 1

def test_mace_workflow_early_convergence(base_config):
    """Test that Step 2 loop breaks early if uncertainty is low."""
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_sg.generate_direct_samples.return_value = generator_mock(5)

    # Oracle returns LOW uncertainty
    mock_oracle = MockOracle(base_config)
    original_compute = mock_oracle.compute_uncertainty
    def low_uncertainty(structures):
        for s in structures:
            s.uncertainty_state = UncertaintyState(gamma_max=0.1, gamma_mean=0.1)
            yield s
    mock_oracle.compute_uncertainty = low_uncertainty # Override

    mock_mace_trainer = MagicMock(spec=Trainer)

    orch = Orchestrator(
        base_config,
        structure_generator=mock_sg,
        oracle=mock_oracle,
        trainer=MagicMock(),
        dynamics_engine=MagicMock(),
        mace_trainer=mock_mace_trainer
    )

    # Mock downstream to avoid crashes even if we skip Step 2 active parts
    with patch('pyacemaker.orchestrator.MaceSurrogateOracle'):
        orch._run_mace_distillation() # Calling internal method to focus test

    # Should verify that we didn't call compute_batch (DFT) because uncertainty was low
    # But wait, logic is: Select -> Check Threshold.
    # If selected < threshold, BREAK.
    # So DFT compute_batch inside the loop should NOT be called.
    # We can check if Oracle's compute_batch was called.
    # MockOracle implements compute_batch but we can spy on it.
    # Actually, let's spy on the trainer. If loop breaks, fine-tuning might be skipped for that iteration.
    # Since we break immediately, mace_trainer.train inside the loop should NOT be called.

    mock_mace_trainer.train.assert_not_called()

def test_mace_workflow_oracle_failure(base_config):
    """Test handling of Oracle failure."""
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_sg.generate_direct_samples.return_value = generator_mock(5)

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

    with pytest.raises(RuntimeError, match="Oracle Failed"):
        orch.run()

def test_config_validation_mace_mode(base_config):
    """Test that invalid configuration prevents running."""
    # Remove MACE config
    base_config.oracle.mace = None

    # Should raise error during initialization or run?
    # Orchestrator init checks oracle types but relies on config flags.
    # If oracle.mace is None, logic in __init__ might pick DFTOracle.
    # But _run_mace_distillation asserts oracle is UncertaintyModel.
    # DFTOracle does not implement UncertaintyModel (in current code).
    # So it should raise TypeError.

    # Setup standard Orchestrator which defaults to DFTOracle if mace is missing
    orch = Orchestrator(base_config)

    # Force enable_mace_distillation = True
    orch.config.distillation.enable_mace_distillation = True

    # The default oracle (DFTOracle) doesn't have compute_uncertainty
    with pytest.raises(TypeError, match="Oracle must implement UncertaintyModel"):
        orch.run()
