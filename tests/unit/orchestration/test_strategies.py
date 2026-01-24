from pathlib import Path
from unittest.mock import MagicMock

from ase import Atoms

from mlip_autopipec.config.schemas.common import EmbeddingConfig
from mlip_autopipec.orchestration.strategies import GammaSelectionStrategy


def test_gamma_selection_strategy():
    # Arrange
    mock_pacemaker = MagicMock()
    mock_pacemaker.select_active_set.return_value = [0, 2] # Select 1st and 3rd

    embedding_config = EmbeddingConfig()
    strategy = GammaSelectionStrategy(mock_pacemaker, embedding_config)

    candidates = [Atoms('H'), Atoms('He'), Atoms('Li')]
    potential_path = Path("fake.yace")

    # Act
    selected = strategy.select(candidates, potential_path)

    # Assert
    assert len(selected) == 2
    assert selected[0].get_chemical_symbols() == ['H']
    assert selected[1].get_chemical_symbols() == ['Li']

    mock_pacemaker.select_active_set.assert_called_once_with(candidates, potential_path)

def test_gamma_selection_strategy_fallback():
    # Arrange
    mock_pacemaker = MagicMock()
    mock_pacemaker.select_active_set.side_effect = Exception("Fail")

    embedding_config = EmbeddingConfig()
    strategy = GammaSelectionStrategy(mock_pacemaker, embedding_config)

    candidates = [Atoms('H'), Atoms('He')]
    potential_path = Path("fake.yace")

    # Act
    selected = strategy.select(candidates, potential_path)

    # Assert
    assert len(selected) == 2 # Should return all on failure
    assert selected == candidates
