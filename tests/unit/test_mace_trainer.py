
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.domain_models.models import Potential, StructureMetadata
from pyacemaker.trainer.mace_trainer import MaceTrainer


class TestMaceTrainer:
    @pytest.fixture
    def config(self, tmp_path):
        mock_config = MagicMock(spec=PYACEMAKERConfig)
        mock_config.oracle = MagicMock()
        mock_config.oracle.mace = MagicMock()
        mock_config.trainer = MagicMock()
        mock_config.project = MagicMock()
        mock_config.project.root_dir = tmp_path
        mock_config.version = "1.0"
        mock_config.distillation = MagicMock()
        mock_config.distillation.step3_mace_finetune.epochs = 10
        return mock_config

    def test_select_active_set_stub(self, config):
        trainer = MaceTrainer(config)
        result = trainer.select_active_set([], 10)
        assert result.selection_criteria == "external_mace_al"
        assert result.structure_ids == []

    def test_train_calls_manager(self, config):
        with patch("pyacemaker.trainer.mace_trainer.MaceManager") as MockManager:
            mock_manager_instance = MockManager.return_value
            # Mock train to return a file path that exists
            dummy_model_path = config.project.root_dir / "dummy.model"
            dummy_model_path.touch()
            mock_manager_instance.train.return_value = dummy_model_path

            trainer = MaceTrainer(config)

            # Create dummy dataset
            from ase import Atoms
            dataset = [
                StructureMetadata(
                    features={"atoms": Atoms("H")}, energy=-1.0, forces=[[0,0,0]]
                )
            ]

            # Mock dataset manager save_iter to just consume
            trainer.dataset_manager = MagicMock()
            trainer.dataset_manager.save_iter = MagicMock()

            result = trainer.train(dataset, epochs=20)

            assert isinstance(result, Potential)
            assert result.path.name.startswith("mace_model_")
            assert result.path.exists()
            assert result.path.parent == config.project.root_dir / "models"

            # Verify manager train called with correct params
            mock_manager_instance.train.assert_called_once()
            args, kwargs = mock_manager_instance.train.call_args
            assert args[2]["max_num_epochs"] == 20
