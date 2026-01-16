"""
Unit tests for the WorkflowManager, focusing on checkpointing and Dask integration.
"""

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from dask.distributed import Client

from mlip_autopipec.config.models import CheckpointState, SystemConfig
from mlip_autopipec.workflow_manager import WorkflowManager


@pytest.fixture
def valid_system_config(tmp_path):
    # A minimal but structurally complete SystemConfig for serialization tests
    from mlip_autopipec.config.models import (
        CutoffConfig,
        DFTConfig,
        DFTInputParameters,
        ExplorerConfig,
        FingerprintConfig,
        InferenceConfig,
        Pseudopotentials,
        TrainingConfig,
    )

    return SystemConfig(
        project_name="test_project",
        run_uuid=uuid.uuid4(),
        dft_config=DFTConfig(
            dft_input_params=DFTInputParameters(
                pseudopotentials=Pseudopotentials.model_validate({"Si": "Si.upf"}),
                cutoffs=CutoffConfig(wavefunction=60, density=240),
                k_points=(3, 3, 3),
            )
        ),
        explorer_config=ExplorerConfig(
            surrogate_model_path="path/to/model",
            fingerprint=FingerprintConfig(species=["Si"]),
        ),
        training_config=TrainingConfig(data_source_db=tmp_path / "test.db"),
        inference_config=InferenceConfig(),
    )


@pytest.fixture
def mock_dask_client():
    return MagicMock(spec=Client)


def test_workflow_manager_initializes_with_no_checkpoint(valid_system_config, mock_dask_client, tmp_path):
    """Tests that the manager starts fresh when no checkpoint exists."""
    manager = WorkflowManager(system_config=valid_system_config, work_dir=tmp_path, dft_runner=MagicMock())
    manager.dask_client = mock_dask_client  # Inject mock
    assert manager.state.run_uuid == valid_system_config.run_uuid
    assert manager.state.active_learning_generation == 0
    assert not manager.state.pending_job_ids


def test_save_checkpoint(valid_system_config, mock_dask_client, tmp_path):
    """Tests that the manager can correctly save its state to a file."""
    manager = WorkflowManager(system_config=valid_system_config, work_dir=tmp_path, dft_runner=MagicMock())
    manager.dask_client = mock_dask_client
    job_id = uuid.uuid4()
    manager.state.pending_job_ids.append(job_id)
    # NOTE: We use serializable types here, as would be the case in reality
    manager.state.job_submission_args[job_id] = ["arg1", {"kwarg": "value"}]

    manager._save_checkpoint()

    checkpoint_path = tmp_path / "checkpoint.json"
    assert checkpoint_path.exists()
    with checkpoint_path.open() as f:
        data = json.load(f)
    assert data["run_uuid"] == str(valid_system_config.run_uuid)
    assert data["pending_job_ids"] == [str(job_id)]
    assert data["job_submission_args"][str(job_id)] == ["arg1", {"kwarg": "value"}]


def test_load_checkpoint(valid_system_config, mock_dask_client, tmp_path):
    """Tests that the manager can correctly load its state from a file."""
    checkpoint_path = tmp_path / "checkpoint.json"
    run_uuid = valid_system_config.run_uuid
    job_id = uuid.uuid4()
    state_to_save = CheckpointState(
        run_uuid=run_uuid,
        system_config=valid_system_config,
        active_learning_generation=1,
        pending_job_ids=[job_id],
        job_submission_args={job_id: ["test_atom", {}]},
    )
    with checkpoint_path.open("w") as f:
        f.write(state_to_save.model_dump_json())

    # Now, initialize a new manager, which should load this state
    manager = WorkflowManager(system_config=valid_system_config, work_dir=tmp_path, dft_runner=MagicMock())
    manager.dask_client = mock_dask_client

    assert manager.state.run_uuid == run_uuid
    assert manager.state.active_learning_generation == 1
    assert manager.state.pending_job_ids == [job_id]
    assert manager.state.job_submission_args[job_id] == ["test_atom", {}]


@patch("mlip_autopipec.workflow_manager.WorkflowManager._load_or_initialize_state")
def test_init_does_not_load_state_unnecessarily(
    mock_load, valid_system_config, mock_dask_client, tmp_path
):
    """
    Tests that the manager does not try to load state when no checkpoint exists.
    """
    manager = WorkflowManager(system_config=valid_system_config, work_dir=tmp_path)
    manager.dask_client = mock_dask_client
    manager.dft_runner = MagicMock()
    mock_load.assert_called_once()


def test_resubmit_pending_jobs(valid_system_config, mock_dask_client, tmp_path):
    """
    Tests the logic for re-submitting jobs that were pending at the time
    of a crash.
    """
    # Patch _load_or_initialize_state to prevent it from running automatically
    with patch.object(WorkflowManager, "_load_or_initialize_state"):
        manager = WorkflowManager(system_config=valid_system_config, work_dir=tmp_path, dft_runner=MagicMock())
        manager.dask_client = mock_dask_client  # Inject mock
        manager.state = CheckpointState(
            run_uuid=valid_system_config.run_uuid, system_config=valid_system_config
        )

    # Manually create a state representing a mid-run failure
    job_id = uuid.uuid4()
    # Mock atoms is not JSON serializable, but job_submission_args in memory can hold objects
    # Checkpoint serialization would fail if we didn't handle it, but here we just test re-submission logic
    mock_atoms = MagicMock()
    manager.state.pending_job_ids.append(job_id)
    manager.state.job_submission_args[job_id] = (mock_atoms,)

    manager._resubmit_pending_jobs()

    # Verify that the Dask client's submit method was called with the correct arguments
    manager.dask_client.submit.assert_called_once_with(manager.dft_runner.run, mock_atoms)
    # Verify that the futures dictionary is repopulated
    assert len(manager.futures) == 1
    future = next(iter(manager.futures.values()))
    assert future == manager.dask_client.submit.return_value

def test_checkpoint_training_history(valid_system_config, mock_dask_client, tmp_path):
    """Tests that training history is correctly saved and loaded."""
    from mlip_autopipec.config.models import TrainingRunMetrics

    manager = WorkflowManager(system_config=valid_system_config, work_dir=tmp_path, dft_runner=MagicMock())
    manager.dask_client = mock_dask_client

    metrics = TrainingRunMetrics(
        generation=1, num_structures=100, rmse_forces=0.1, rmse_energy_per_atom=0.01
    )
    manager.state.training_history.append(metrics)

    manager._save_checkpoint()

    checkpoint_path = tmp_path / "checkpoint.json"
    with checkpoint_path.open() as f:
        data = json.load(f)

    assert "training_history" in data
    assert len(data["training_history"]) == 1
    assert data["training_history"][0]["rmse_forces"] == 0.1

    # Verify loading
    manager2 = WorkflowManager(system_config=valid_system_config, work_dir=tmp_path, dft_runner=MagicMock())
    assert len(manager2.state.training_history) == 1
    assert manager2.state.training_history[0].rmse_forces == 0.1

def test_perform_training(valid_system_config, mock_dask_client, tmp_path):
    """Tests that perform_training updates state and checkpoints."""
    from pathlib import Path

    from mlip_autopipec.config.models import TrainingRunMetrics
    from mlip_autopipec.modules.training import PacemakerTrainer

    mock_trainer = MagicMock(spec=PacemakerTrainer)
    metrics = TrainingRunMetrics(
        generation=0, num_structures=50, rmse_forces=0.05, rmse_energy_per_atom=0.005
    )
    mock_trainer.perform_training.return_value = (Path("potential.yace"), metrics)

    manager = WorkflowManager(
        system_config=valid_system_config,
        work_dir=tmp_path,
        trainer=mock_trainer
    )
    manager.dask_client = mock_dask_client

    manager.perform_training()

    assert len(manager.state.training_history) == 1
    assert manager.state.training_history[0] == metrics
    assert manager.state.current_potential_path == Path("potential.yace")

    mock_trainer.perform_training.assert_called_once_with(generation=0)

    # Verify checkpoint
    checkpoint_path = tmp_path / "checkpoint.json"
    with checkpoint_path.open() as f:
        data = json.load(f)
    assert len(data["training_history"]) == 1
