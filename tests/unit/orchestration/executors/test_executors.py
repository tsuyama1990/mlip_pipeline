"""
Tests for Phase Executors.
"""

import pytest
from unittest.mock import MagicMock, patch, ANY
from mlip_autopipec.orchestration.executors.exploration_executor import ExplorationExecutor
from mlip_autopipec.orchestration.executors.dft_executor import DFTExecutor
from mlip_autopipec.orchestration.executors.training_executor import TrainingExecutor
from mlip_autopipec.orchestration.executors.inference_executor import InferenceExecutor
from mlip_autopipec.orchestration.phase_executor import PhaseExecutor

@pytest.fixture
def mock_manager():
    manager = MagicMock()
    manager.config = MagicMock()
    manager.db_manager = MagicMock()
    manager.task_queue = MagicMock()
    manager.builder = MagicMock()
    manager.surrogate = MagicMock()
    manager.work_dir = MagicMock()
    manager.state.current_generation = 1
    return manager

def test_exploration_executor(mock_manager):
    executor = ExplorationExecutor(mock_manager)

    # Mock builder output
    atoms = MagicMock()
    executor._builder.build.return_value = [atoms]

    # Mock config
    mock_manager.config.surrogate_config = None

    result = executor.execute()

    assert result is True
    mock_manager.db_manager.save_candidate.assert_called()

def test_dft_executor(mock_manager):
    executor = DFTExecutor(mock_manager)

    # Mock DB entries
    atoms = MagicMock()
    mock_manager.db_manager.select_entries.return_value = [(1, atoms)]

    # Mock QERunner
    # We need to mock _create_qe_runner since it depends on dft_config being valid
    with patch.object(executor, '_create_qe_runner') as mock_create_runner:
        mock_runner = mock_create_runner.return_value

        # Mock Queue
        mock_manager.task_queue.wait_for_completion.return_value = [MagicMock()] # Success result

        result = executor.execute()

        assert result is True
        mock_manager.db_manager.save_dft_result.assert_called()

def test_training_executor(mock_manager):
    executor = TrainingExecutor(mock_manager)

    # Mock DatasetBuilder (import path inside executor module or where it's used)
    # The executor imports it from mlip_autopipec.training.dataset
    with patch("mlip_autopipec.orchestration.executors.training_executor.DatasetBuilder") as MockBuilder, \
         patch.object(executor, '_create_pacemaker_wrapper') as mock_create_wrapper:

        mock_wrapper = mock_create_wrapper.return_value
        result_obj = MagicMock()
        result_obj.success = True
        result_obj.potential_path = "dummy.yace"
        mock_wrapper.train.return_value = result_obj

        # Mock shutil.copy2
        with patch("mlip_autopipec.orchestration.executors.training_executor.shutil.copy2"):
            result = executor.execute()

        assert result is True
        mock_wrapper.train.assert_called()

def test_inference_executor(mock_manager):
    executor = InferenceExecutor(mock_manager)

    # Mock DB selection for start atoms
    start_atoms = MagicMock()
    start_atoms.get_chemical_symbols.return_value = ["H"]

    # Generator for select
    def atom_gen():
        yield start_atoms
    mock_manager.db_manager.select.return_value = atom_gen()

    # Mock potential existence
    # work_dir is a MagicMock. work_dir / "current.yace" returns another MagicMock.
    mock_manager.work_dir.__truediv__.return_value.exists.return_value = True

    # Mock LammpsRunner creation
    # Use string path if executor is imported as module
    with patch.object(executor, '_create_lammps_runner') as mock_create_runner:
        mock_runner = mock_create_runner.return_value

        # Scenario: High uncertainty
        res_obj = MagicMock()
        res_obj.uncertain_structures = ["dump.out"]
        mock_runner.run.return_value = res_obj

        # Mock EmbeddingExtractor and ase.io.read
        # ase.io.read is imported inside the method, so we patch where it is used in the module
        with patch("mlip_autopipec.orchestration.executors.inference_executor.EmbeddingExtractor") as MockExtractor, \
             patch("mlip_autopipec.orchestration.executors.inference_executor.read") as mock_read:

            # Mock read returning frames
            frame = MagicMock()
            # Need strict dictionary emulation for arrays
            gamma_array = MagicMock()
            gamma_array.argmax.return_value = 0
            # Ensure proper casting to int
            gamma_array.argmax.return_value.__int__.return_value = 0

            frame.arrays = {'c_gamma': gamma_array, 'type': [1]}
            mock_read.return_value = [frame]

            # Mock Extraction
            MockExtractor.return_value.extract.return_value = MagicMock()

            # Fix Potential Existence Check
            # `execute_inference` does `potential_path = self.manager.work_dir / "current.yace"`
            # Then `if not potential_path.exists(): ...`
            # Since `work_dir` is a MagicMock, `__truediv__` returns a new MagicMock by default.
            # We must configure that SPECIFIC return value.

            potential_path_mock = mock_manager.work_dir / "current.yace"
            potential_path_mock.exists.return_value = True

            result = executor.execute()

            # If still fails, debug output
            if not result:
                print("DEBUG: execute returned False. Potential Exists:", potential_path_mock.exists())

            assert result is True
            mock_manager.db_manager.save_candidate.assert_called()

def test_phase_executor_facade(mock_manager):
    # Test that PhaseExecutor correctly delegates
    facade = PhaseExecutor(mock_manager)

    with patch.object(facade.exploration, 'execute') as mock_expl:
        facade.execute_exploration()
        mock_expl.assert_called_once()
