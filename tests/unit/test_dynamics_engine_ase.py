"""Unit tests for ASE Dynamics Engine."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.domain_models.models import (
    Potential,
    PotentialType,
    StructureMetadata,
    UncertaintyState,
)
from pyacemaker.modules.dynamics_engine import ASEDynamicsEngine


@pytest.fixture
def mock_potential() -> Potential:
    """Create a mock potential."""
    return Potential(
        path=Path("mock.model"),
        type=PotentialType.MACE,
        version="1.0",
        metrics={},
        parameters={},
    )


@patch("pyacemaker.modules.dynamics_engine.ASEDynamicsEngine._run_md_single_seed")
def test_ase_dynamics_engine_exploration(
    mock_run_md: MagicMock,
    full_config: PYACEMAKERConfig,
    mock_potential: Potential,
) -> None:
    """Test run_exploration iterates over seeds."""
    engine = ASEDynamicsEngine(full_config)

    seeds = [StructureMetadata(id=uuid4()), StructureMetadata(id=uuid4())]

    # Mock return values (lists now)
    mock_run_md.side_effect = [
        [
            StructureMetadata(
                id=uuid4(),
                tags=["halted"],
                uncertainty_state=UncertaintyState(gamma_max=2.0),
            )
        ],
        [],  # Second seed didn't yield high uncertainty
    ]

    results = list(engine.run_exploration(mock_potential, seeds))

    assert len(results) == 1
    assert results[0].tags == ["halted"]
    assert mock_run_md.call_count == 2


@patch("pyacemaker.modules.dynamics_engine.ASEDynamicsEngine._run_md_single_seed")
def test_ase_dynamics_engine_exploration_empty(
    mock_run_md: MagicMock,
    full_config: PYACEMAKERConfig,
    mock_potential: Potential,
) -> None:
    """Test run_exploration with no seeds."""
    engine = ASEDynamicsEngine(full_config)
    results = list(engine.run_exploration(mock_potential, []))
    assert len(results) == 0
    mock_run_md.assert_not_called()


def test_ase_dynamics_engine_production_not_implemented(
    full_config: PYACEMAKERConfig,
) -> None:
    """Test run_production raises NotImplementedError."""
    engine = ASEDynamicsEngine(full_config)
    with pytest.raises(NotImplementedError):
        engine.run_production(MagicMock())
