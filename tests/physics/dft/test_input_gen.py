import pytest
from pathlib import Path
from ase import Atoms

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.calculation import DFTConfig
from mlip_autopipec.physics.dft.input_gen import InputGenerator


@pytest.fixture
def sample_structure():
    atoms = Atoms(
        symbols=["Si", "Si"],
        positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
        cell=[[2.7, 0, 0], [0, 2.7, 0], [0, 0, 2.7]],
        pbc=True,
    )
    return Structure.from_ase(atoms)


@pytest.fixture
def dft_config():
    return DFTConfig(
        command="pw.x",
        pseudopotentials={"Si": Path("Si.upf")},
        ecutwfc=30.0,
        kspacing=0.2,  # 2*pi / (2.7 * 0.2) ~= 6.28 / 0.54 ~= 11.6 -> 12
        smearing="gaussian",
        degauss=0.02,
    )


def test_generate_input_kpoints(sample_structure, dft_config):
    generator = InputGenerator()
    content = generator.generate_input(sample_structure, dft_config)

    # Check K-points
    # L=2.7, kspacing=0.2 -> N ~ 12
    assert "K_POINTS automatic" in content
    # Allow variable spaces
    assert "12 12 12 0 0 0" in " ".join(content.split())


def test_generate_input_control_flags(sample_structure, dft_config):
    generator = InputGenerator()
    content = generator.generate_input(sample_structure, dft_config)

    # Remove whitespace for easier checking
    normalized = content.replace(" ", "").replace("\t", "")
    assert "tprnfor=.true." in normalized
    assert "tstress=.true." in normalized
    assert "ecutwfc=30.0" in normalized
    assert "Si.upf" in normalized
    assert "ATOMIC_POSITIONSangstrom" in normalized.replace("\n", "")
