import numpy as np
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.config import StructureGenConfig
from mlip_autopipec.modules.structure_gen.generator import StructureGenerator
from mlip_autopipec.modules.structure_gen.strategies import ColdStartStrategy, RattleStrategy

def test_cold_start_strategy() -> None:
    strategy = ColdStartStrategy()
    # ASE diamond build gives 8 atoms for cubic=True which is default in some contexts but build.bulk might return primitive (2 atoms).
    # We should ensure we use cubic=True in implementation if we want 8 atoms.
    # Let's assume implementation uses cubic=True or handles it.
    s = strategy.generate(element="Si", crystal_structure="diamond", lattice_constant=5.43, supercell=(1,1,1))

    # Check symbol count. Primitive diamond has 2 atoms. Cubic has 8.
    # If we assume build.bulk("Si", "diamond", ...) returns primitive by default, then 2.
    # If we implement it to return cubic, then 8.
    # Spec "One-Shot Pipeline" mentioned "StructureBuilder... build_initial_structure('Si')... 8 atoms for cubic".
    # So I will implement cubic=True.
    assert len(s.symbols) == 8

def test_rattle_strategy() -> None:
    s = Structure(
        symbols=["Si"], positions=np.zeros((1,3)), cell=np.eye(3), pbc=(True,True,True)
    )
    strategy = RattleStrategy(stdev=0.1, seed=42)
    rattled = strategy.apply(s)
    assert not np.array_equal(s.positions, rattled.positions)
    assert np.allclose(s.cell, rattled.cell)

def test_structure_generator() -> None:
    config = StructureGenConfig(
        element="Si",
        crystal_structure="diamond",
        lattice_constant=5.43,
        supercell=(1,1,1),
        rattle_stdev=0.01
    )
    # Generator takes config and seed?
    # Spec says StructureBuilder ensures deterministic structure generation.
    gen = StructureGenerator(config, seed=123)
    s = gen.build()
    assert len(s.symbols) == 8
    assert isinstance(s, Structure)
