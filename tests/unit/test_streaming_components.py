"""Unit tests for streaming components logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.config import ProjectConfig, PYACEMAKERConfig
from pyacemaker.domain_models.models import StructureMetadata
from pyacemaker.orchestrator import Orchestrator


@pytest.fixture
def streaming_config(tmp_path: Path) -> PYACEMAKERConfig:
    return PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="streaming_test", root_dir=tmp_path),
        orchestrator={
            "max_cycles": 1,
            "dataset_file": "dataset.pckl.gzip",
            "validation_file": "val.pckl.gzip",
            "validation_buffer_size": 2,
        },
        oracle={"dft": {"pseudopotentials": {"Fe": "mock.pbe"}}},
        distillation={
            "enable_mace_distillation": False,
            "step4_surrogate_sampling": {"target_points": 1000},
            "step7_pacemaker_finetune": {"enable": True}
        }
    )


def test_orchestrator_training_phase_streaming(streaming_config: PYACEMAKERConfig) -> None:
    """Verify StandardActiveLearningWorkflow._run_training_phase uses streaming."""
    orchestrator = Orchestrator(config=streaming_config)
    workflow = orchestrator.standard_workflow

    # Mock trainer
    workflow.trainer = MagicMock()

    # Create directories/files
    workflow.training_path.parent.mkdir(parents=True, exist_ok=True)
    workflow.dataset_path.touch()
    workflow.validation_path.touch()
    workflow.training_path.touch()

    # Mock Splitter to avoid real IO
    with patch("pyacemaker.workflows.active_learning.DatasetSplitter") as mock_splitter_cls:
        mock_splitter = mock_splitter_cls.return_value

        # Must return valid StructureMetadata with atoms for metadata_to_atoms
        s = StructureMetadata(features={"atoms": Atoms("H")})
        mock_splitter.train_stream.return_value = iter([s])
        mock_splitter.processed_count = 1

        workflow._run_training_phase()

        assert mock_splitter.train_stream.called
        assert workflow.trainer.train.called
