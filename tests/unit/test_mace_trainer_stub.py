
import pytest
from unittest.mock import MagicMock
from pyacemaker.trainer.mace_trainer import MaceTrainer
from pyacemaker.core.config import PYACEMAKERConfig, TrainerConfig

class TestMaceTrainerStub:
    @pytest.fixture
    def config(self):
        mock_config = MagicMock(spec=PYACEMAKERConfig)
        mock_config.oracle = MagicMock()
        mock_config.oracle.mace = MagicMock()
        mock_config.trainer = MagicMock() # Needed for BaseTrainer init
        mock_config.version = "1.0"
        return mock_config

    def test_select_active_set_stub(self, config):
        trainer = MaceTrainer(config)
        result = trainer.select_active_set([], 10)
        assert result.selection_criteria == "external_mace_al"
        assert result.structure_ids == []
