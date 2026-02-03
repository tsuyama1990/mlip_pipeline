from pathlib import Path

from ase import Atoms

# We import the class we intend to create.
# This will fail until the file is created, which is expected in TDD.
from mlip_autopipec.physics.dynamics.input_generator import LammpsInputGenerator


def test_hybrid_pair_style_generation() -> None:
    # Setup
    # Elements: Ti (Z=22), O (Z=8)
    atoms = Atoms("TiO", positions=[[0, 0, 0], [1.5, 1.5, 1.5]])
    potential_path = Path("test.yace")
    generator = LammpsInputGenerator(potential_path)

    # Act
    # We pass a dummy data_file name that would correspond to the structure
    input_str = generator.generate_input(atoms, data_file="data.lammps", parameters={"temp": 300})

    # Assert
    # 1. Check pair_style
    assert "pair_style hybrid/overlay pace" in input_str
    assert "zbl" in input_str

    # 2. Check pair_coeff for PACE
    # The generator should extract elements from atoms and put them in the line
    # Expect: pair_coeff * * pace test.yace
    assert "pair_coeff * * pace test.yace" in input_str

    # 3. Check pair_coeff for ZBL
    # ZBL parameters: Z_i Z_j cutoff_inner cutoff_outer
    assert "pair_coeff" in input_str
    assert "zbl" in input_str


def test_watchdog_fix() -> None:
    atoms = Atoms("Al")
    generator = LammpsInputGenerator(Path("test.yace"))
    input_str = generator.generate_input(atoms, data_file="data.lammps", parameters={"temp": 300, "gamma_threshold": 5.0})

    # Check for fix halt
    assert "fix watchdog all halt" in input_str
    assert "v_max_gamma > 5.0" in input_str
    assert "error hard" in input_str
