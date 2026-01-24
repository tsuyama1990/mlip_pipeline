import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.exceptions import GeneratorError
from mlip_autopipec.generator.transformations import apply_rattle, apply_strain


@pytest.fixture
def simple_atoms() -> Atoms:
    return Atoms("Al", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)


def test_apply_strain(simple_atoms: Atoms) -> None:
    strain = np.diag([0.1, 0.1, 0.1])  # 10% expansion
    strained = apply_strain(simple_atoms, strain)

    # Original cell was 3x3x3. New cell should be 3.3x3.3x3.3
    expected_cell = np.diag([3.3, 3.3, 3.3])
    np.testing.assert_allclose(strained.get_cell(), expected_cell)

    # Positions should scale? apply_strain implementation uses set_cell(scale_atoms=True)
    # Original pos [0,0,0] -> [0,0,0]
    np.testing.assert_allclose(strained.positions, [[0, 0, 0]])

    assert strained.info["config_type"] == "strain"
    assert "strain_tensor" in strained.info


def test_apply_strain_invalid_input(simple_atoms: Atoms) -> None:
    with pytest.raises(GeneratorError):
        apply_strain(simple_atoms, np.array([0.1, 0.1]))  # Invalid shape


def test_apply_rattle(simple_atoms: Atoms) -> None:
    sigma = 0.1
    rng = np.random.default_rng(42)
    rattled = apply_rattle(simple_atoms, sigma, rng=rng)

    assert rattled.info["config_type"] == "rattle"
    assert rattled.info["rattle_sigma"] == sigma

    # Positions should change
    assert not np.allclose(rattled.positions, simple_atoms.positions)

    # Check deterministic behavior
    rng2 = np.random.default_rng(42)
    rattled2 = apply_rattle(simple_atoms, sigma, rng=rng2)
    np.testing.assert_allclose(rattled.positions, rattled2.positions)


def test_apply_rattle_invalid_sigma(simple_atoms: Atoms) -> None:
    with pytest.raises(GeneratorError):
        apply_rattle(simple_atoms, -0.1)
