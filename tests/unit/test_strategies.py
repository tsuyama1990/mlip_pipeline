import numpy as np

from mlip_autopipec.domain_models.config import StructureGenConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.modules.structure_gen.strategies import (
    ColdStartStrategy,
    RattleStrategy,
)


def test_cold_start_strategy() -> None:
    """Test ColdStartStrategy generates correct structure."""
    config = StructureGenConfig(element="Si", lattice_constant=5.43)
    strat = ColdStartStrategy()
    s = strat.generate(config)
    assert isinstance(s, Structure)
    assert "Si" in s.symbols


def test_rattle_strategy() -> None:
    """Test RattleStrategy perturbs positions."""
    config = StructureGenConfig(element="Si")
    s_init = ColdStartStrategy().generate(config)
    strat = RattleStrategy()

    # stdev > 0 should change positions
    s_rattled = strat.apply(s_init, stdev=0.1, seed=42)
    assert isinstance(s_rattled, Structure)
    assert not np.allclose(s_init.positions, s_rattled.positions)

    # stdev = 0 should not change positions
    s_clean = strat.apply(s_init, stdev=0.0)
    assert np.allclose(s_init.positions, s_clean.positions)
