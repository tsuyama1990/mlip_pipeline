"""UAT tests for Cycle 04 (Surrogate Labeling & Pacemaker Training)."""

from unittest.mock import MagicMock

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from pyacemaker.core.interfaces import Oracle, UncertaintyModel
from pyacemaker.domain_models.models import (
    Potential,
    PotentialType,
    StructureStatus,
)
from pyacemaker.modules.mace_workflow import MaceDistillationWorkflow
from pyacemaker.oracle.dataset import DatasetManager


class MockOracle(Oracle, UncertaintyModel):
    def compute_batch(self, structures): pass
    def compute_uncertainty(self, structures): pass
    def run(self): pass
    def update_model(self, path): pass


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration."""
    config = MagicMock()
    config.project = MagicMock()
    config.project.root_dir = tmp_path
    config.version = "1.0.0"
    config.distillation = MagicMock()
    config.distillation.surrogate_dataset_file = "surrogate_dataset.pckl.gzip"
    # Mock trainer config
    config.trainer.mock = True
    config.oracle.mock = True
    return config


@pytest.fixture
def mock_workflow(mock_config, tmp_path):
    """Create a MaceDistillationWorkflow with mocked dependencies."""
    dataset_manager = DatasetManager()
    oracle = MagicMock(spec=MockOracle)
    trainer = MagicMock()
    mace_trainer = MagicMock()
    dynamics_engine = MagicMock()
    structure_generator = MagicMock()

    return MaceDistillationWorkflow(
        config=mock_config,
        dataset_manager=dataset_manager,
        dataset_path=tmp_path / "dataset.pckl.gzip",
        oracle=oracle,
        trainer=trainer,
        mace_trainer=mace_trainer,
        dynamics_engine=dynamics_engine,
        structure_generator=structure_generator,
        validation_path=tmp_path / "validation.pckl.gzip",
        training_path=tmp_path / "training.pckl.gzip",
    )


def test_scenario_01_successful_batch_labeling(mock_workflow, tmp_path):
    """Scenario 01: Successful Batch Labeling."""
    # GIVEN a list of unlabeled structures (surrogate_candidates)
    # Created as a file to simulate Step 4 output
    surrogate_path = tmp_path / "surrogate_unlabeled.pckl.gzip"

    atoms_list = [Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]) for _ in range(5)]
    # Save using dataset manager
    mock_workflow.dataset_manager.save_iter(iter(atoms_list), surrogate_path)

    # AND a fine-tuned MaceSurrogateOracle (mocked)
    # We mock compute_batch to simulate labeling
    def mock_compute_batch(stream):
        for s in stream:
            s.energy = -10.0
            s.forces = [[0.0, 0.0, 0.0]] * len(s.features["atoms"])
            s.status = StructureStatus.CALCULATED
            yield s

    mock_workflow.oracle.compute_batch.side_effect = mock_compute_batch

    # WHEN the Orchestrator executes Step 5 (via workflow)
    # We call the internal step method directly for UAT
    dataset_path = mock_workflow._step5_surrogate_labeling(surrogate_path)

    # THEN it should run batch prediction on the structures
    mock_workflow.oracle.compute_batch.assert_called_once()

    # AND it should save the labeled structures to surrogate_dataset.pckl.gzip
    assert dataset_path.exists()
    assert dataset_path.stat().st_size > 0  # Verify file is not empty
    assert dataset_path.name == "surrogate_dataset.pckl.gzip"

    # AND the dataset should contain valid energy and force values
    loaded_data = list(mock_workflow.dataset_manager.load_iter(dataset_path))
    assert len(loaded_data) == 5
    for atoms in loaded_data:
        assert atoms.info["energy"] == -10.0
        assert len(atoms.get_forces()) == 2


def test_scenario_02_base_ace_training(mock_workflow, tmp_path):
    """Scenario 02: Base ACE Training."""
    # GIVEN a surrogate_dataset.pckl.gzip
    dataset_path = tmp_path / "surrogate_dataset.pckl.gzip"
    atoms = Atoms("H", positions=[[0, 0, 0]])
    # Attach results properly
    atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=[[0, 0, 0]])
    atoms_list = [atoms]
    mock_workflow.dataset_manager.save_iter(iter(atoms_list), dataset_path)

    # AND a PacemakerTrainer configuration (mocked in workflow init)
    mock_workflow.trainer.train.return_value = Potential(
        path=tmp_path / "potential.yace",
        type=PotentialType.PACE,
        version="1.0",
        metrics={},
        parameters={}
    )

    # WHEN the Orchestrator executes Step 6
    potential = mock_workflow._step6_pacemaker_base_training(dataset_path)

    # THEN it should generate a valid input.yaml (Implied by Trainer.train logic)
    # AND it should invoke the training process
    mock_workflow.trainer.train.assert_called_once()

    # AND it should produce a potential.yace file (returned object)
    assert potential.path == tmp_path / "potential.yace"
    assert potential.type == PotentialType.PACE


def test_scenario_03_error_handling(mock_workflow, tmp_path):
    """Scenario 03: Error Handling in Workflow."""
    # GIVEN a workflow where Oracle fails
    mock_workflow.oracle.compute_batch.side_effect = Exception("Oracle failure")

    # We mock earlier steps to return valid paths so we reach labeling/calculation
    mock_workflow._step1_direct_sampling = MagicMock(return_value=tmp_path / "pool.pckl")
    mock_workflow._step2_active_learning_loop = MagicMock(return_value=None)
    mock_workflow._step4_surrogate_data_generation = MagicMock(return_value=tmp_path / "surrogate.pckl")

    # Need to mock the file existence for step 5 to proceed to load stream
    (tmp_path / "surrogate.pckl").touch()

    # WHEN running the workflow
    result = mock_workflow.run()

    # THEN it should return a failure result
    assert result.status == "failed"
    assert "Oracle failure" in str(result.error)
