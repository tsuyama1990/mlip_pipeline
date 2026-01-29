"""Unit tests for structure generation."""

import pytest
import numpy as np
from ase import Atoms
from mlip_autopipec.domain_models.config import ExplorationConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.modules.structure_gen.strategies import ColdStartStrategy, RandomPerturbationStrategy
from mlip_autopipec.modules.structure_gen.generator import StructureGenerator

def test_cold_start_strategy() -> None:
    """Test cold start strategy generates valid structures."""
    config = ExplorationConfig(
        strategy="template",
        supercell_size=[1, 1, 1],
        rattle_amplitude=0.0,
        num_candidates=1,
        composition="Si"
    )
    strategy = ColdStartStrategy()
    structures = strategy.generate(config)

    assert len(structures) >= 1
    assert isinstance(structures[0], Structure)
    # Check that we have Si
    assert "Si" in structures[0].formatted_formula

def test_random_perturbation_strategy() -> None:
    """Test perturbation strategy."""
    config = ExplorationConfig(
        strategy="random",
        rattle_amplitude=0.5
    )

    # Create a simple structure
    atoms = Atoms('Si', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    structure = Structure.from_ase(atoms)

    strategy = RandomPerturbationStrategy()
    perturbed = strategy.apply(structure, config)

    assert isinstance(perturbed, Structure)
    assert len(perturbed.positions) == len(structure.positions)

    # Check if positions moved (rattle > 0)
    # With 0.5 amplitude, it is extremely unlikely to remain exactly at 5.0
    pos_orig = np.array(structure.positions)
    pos_new = np.array(perturbed.positions)
    assert not np.allclose(pos_orig, pos_new)

def test_generator_integration() -> None:
    """Test generator orchestrating the strategies."""
    config = ExplorationConfig(
        strategy="random",
        num_candidates=2,
        composition="Al"
    )
    generator = StructureGenerator(config)
    candidates = generator.generate_initial_set()

    assert len(candidates) == 2
    assert candidates[0].source == "cold_start"
    assert candidates[0].status == "PENDING"

def test_generator_apply_strategy() -> None:
    """Test applying strategy via generator."""
    config = ExplorationConfig(strategy="random", rattle_amplitude=0.1)
    generator = StructureGenerator(config)

    # Mock structure
    atoms = Atoms('Si', positions=[[0,0,0]], cell=[10,10,10], pbc=True)
    struct = Structure.from_ase(atoms)

    candidates = generator.apply_strategy([struct], strategy_name="random")

    assert len(candidates) == 1
    assert candidates[0].source == "random_perturbation"
    assert candidates[0].priority == 0.5

def test_cold_start_no_composition() -> None:
    """Test cold start with missing composition."""
    config = ExplorationConfig(strategy="template", composition=None)
    strategy = ColdStartStrategy()
    structures = strategy.generate(config)
    assert len(structures) == 0
