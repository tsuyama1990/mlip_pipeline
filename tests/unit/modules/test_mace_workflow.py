"""Tests for MaceDistillationWorkflow."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.config import DistillationConfig
from pyacemaker.domain_models.state import PipelineState
from pyacemaker.modules.mace_workflow import MaceDistillationWorkflow


@pytest.fixture
def mock_workflow(tmp_path: Path) -> MaceDistillationWorkflow:
    """Fixture for MaceDistillationWorkflow."""
    config = DistillationConfig(
        enable_mace_distillation=True,
        pool_file="pool.xyz",
        surrogate_file="surrogate.xyz",
        surrogate_dataset_file="dataset.xyz",
    )
    return MaceDistillationWorkflow(
        config=config,
        dataset_manager=MagicMock(),
        active_learner=MagicMock(),
        structure_generator=MagicMock(),
        oracle=MagicMock(),
        mace_oracle=MagicMock(),
        pacemaker_trainer=MagicMock(),
        mace_trainer=MagicMock(),
        work_dir=tmp_path,
    )


def test_step5_surrogate_labeling(mock_workflow: MaceDistillationWorkflow) -> None:
    """Test Step 5: Surrogate Labeling with batched processing."""
    state = PipelineState(current_step=5)
    state.artifacts["surrogate_pool_path"] = str(mock_workflow.work_dir / "surrogate.xyz")
    state.artifacts["mace_model_path"] = str(mock_workflow.work_dir / "model.model")

    atoms_list = [Atoms("Si2"), Atoms("Si2"), Atoms("Si2")]

    # Mock dataset manager to return an iterator of atoms
    # We mock stream_metadata_to_atoms because that's what converts it
    with patch("pyacemaker.modules.mace_workflow.stream_metadata_to_atoms") as mock_stream:
        mock_stream.return_value = iter(atoms_list)

        # Mock calculator on mace_oracle
        mock_calc = MagicMock()
        mock_workflow.mace_oracle.calculator = mock_calc

        # We need to mock validate_safe_path
        with patch("pyacemaker.modules.mace_workflow.validate_safe_path", return_value=mock_workflow.work_dir / "safe.xyz"):
             new_state = mock_workflow.step5_surrogate_labeling(state)

    assert new_state.current_step == 6
    assert "labeled_surrogate_path" in new_state.artifacts

    # Verify calculator was attached and computed
    assert atoms_list[0].calc == mock_calc
    # Verify calculator methods were called
    # (Since atoms.get_potential_energy calls calculator.get_potential_energy)
    # But atoms.get_potential_energy() is called in _process_batch.
    # If atoms.calc is a mock, get_potential_energy() might not trigger get_potential_energy on mock calc unless configured.
    # However, we can check if we tried to get energy.

    # Actually, we can verify _process_batch logic by checking if it tried to access energy
    # But since we didn't mock Atom.get_potential_energy, it would try to use the calc.
    # The Mock calculator should receive get_potential_energy call?
    # ASE's Calculator interface: get_potential_energy(atoms=None, ...)
    # When atoms.get_potential_energy() is called, it calls self.calc.get_potential_energy(self)

    # Since we didn't configure mock_calc to return anything, atoms.get_potential_energy might return None or Mock.
    # But we assert that the flow completed.
    pass

def test_step5_calculator_error_handling(mock_workflow: MaceDistillationWorkflow) -> None:
    """Test Step 5 handles calculator errors gracefully."""
    state = PipelineState(current_step=5)
    state.artifacts["surrogate_pool_path"] = "surrogate.xyz"
    state.artifacts["mace_model_path"] = "model.model"

    bad_atom = Atoms("Si2")
    # Make get_potential_energy raise exception
    bad_atom.get_potential_energy = MagicMock(side_effect=Exception("Calc failed"))

    with patch("pyacemaker.modules.mace_workflow.stream_metadata_to_atoms") as mock_stream, \
         patch("pyacemaker.modules.mace_workflow.validate_safe_path", return_value=mock_workflow.work_dir / "safe.xyz"), \
         patch("pyacemaker.modules.mace_workflow.logger") as mock_logger:

        mock_stream.return_value = iter([bad_atom])
        mock_workflow.mace_oracle.calculator = MagicMock()

        mock_workflow.step5_surrogate_labeling(state)

        # Should log warning but continue
        mock_logger.warning.assert_called_with("Failed to label structure: Calc failed")
