from pathlib import Path
import pytest
import numpy as np
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.calculation import DFTConfig
# We assume InputGenerator will be in mlip_autopipec.physics.dft.input_gen
from mlip_autopipec.physics.dft.input_gen import InputGenerator

@pytest.fixture
def simple_structure():
    return Structure(
        symbols=["Si", "Si"],
        positions=np.array([[0.0, 0.0, 0.0], [1.3, 1.3, 1.3]]),
        cell=np.array([[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]),
        pbc=(True, True, True)
    )

@pytest.fixture
def dft_config():
    return DFTConfig(
        command="pw.x",
        pseudopotentials={"Si": Path("Si.UPF")},
        ecutwfc=30.0,
        kspacing=0.2 # Rough grid
    )

def test_generate_input_string(simple_structure, dft_config):
    # kspacing 0.2, L=5.43. N = 2*pi / (5.43 * 0.2) = 6.28 / 1.086 = 5.7 -> 6
    # So K_POINTS should be 6 6 6

    content = InputGenerator.generate(simple_structure, dft_config)

    assert "ATOMIC_SPECIES" in content
    assert "Si.UPF" in content
    assert "Si " in content
    assert "K_POINTS automatic" in content
    # Allow for variations in spacing
    assert "6 6 6" in content
    assert "tprnfor" in content and ".true." in content
    assert "tstress" in content and ".true." in content
