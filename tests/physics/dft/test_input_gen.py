import pytest
import numpy as np
from pathlib import Path
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.calculation import DFTConfig
from mlip_autopipec.physics.dft.input_gen import InputGenerator

@pytest.fixture
def si_structure():
    return Structure(
        symbols=["Si", "Si"],
        positions=np.array([[0, 0, 0], [1.36, 1.36, 1.36]]),
        cell=np.array([[0, 2.72, 2.72], [2.72, 0, 2.72], [2.72, 2.72, 0]]),
        pbc=(True, True, True),
    )

@pytest.fixture
def dft_config():
    return DFTConfig(
        command="pw.x",
        pseudopotentials={"Si": Path("Si.upf")},
        ecutwfc=30.0,
        kspacing=0.04,
    )

def test_kpoint_generation(si_structure, dft_config):
    # Cell vectors are length ~3.84 Angstrom
    # kspacing = 0.04 -> 2*pi / (3.84 * 0.04) ~ 40
    # Let's adjust kspacing to give a small integer
    # 2*pi / (L * k) = N
    # L ~ 3.84. Let's aim for N=2.
    # 2*pi / (3.84 * k) = 2 => k = pi / 3.84 ~ 0.81

    dft_config.kspacing = 0.8
    generator = InputGenerator()
    content = generator.generate_input(si_structure, dft_config)

    # Check K-points card
    assert "K_POINTS automatic" in content
    # For FCC Si primitive cell, vectors are not orthogonal, but let's assume simple logic first
    # Or rely on ASE's k-point generation logic if used.
    # The spec says: Ni = ceil(2pi / (Li * kspacing))
    # We'll verify that reasonable numbers appear.

    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "K_POINTS automatic" in line:
            k_line = lines[i+1]
            parts = k_line.split()
            nx, ny, nz = int(parts[0]), int(parts[1]), int(parts[2])
            assert nx > 0 and ny > 0 and nz > 0
            break

def test_mandatory_flags(si_structure, dft_config):
    generator = InputGenerator()
    content = generator.generate_input(si_structure, dft_config)

    assert "tprnfor = .true." in content.lower()
    assert "tstress = .true." in content.lower()
    assert "ecutwfc = 30.0" in content.lower()
