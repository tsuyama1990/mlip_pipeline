"""Fixtures for the test suite."""

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.config.system import DFTParams, Pseudopotentials, SystemConfig


@pytest.fixture
def sample_system_config() -> SystemConfig:
    """Provide a sample SystemConfig for a Nickel calculation."""
    dft_params = DFTParams(
        pseudopotentials=Pseudopotentials(root={"Ni": "Ni.pbe-n-rrkjus_psl.1.0.0.UPF"}),
        system={
            "nat": 1,
            "ntyp": 1,
            "ecutwfc": 60.0,
            "nspin": 2,
        },
    )
    return SystemConfig(dft=dft_params, db_path="test.db")


@pytest.fixture
def sample_atoms() -> Atoms:
    """Provide a sample single-atom ASE Atoms object."""
    atoms = Atoms("Ni", positions=[(0, 0, 0)], cell=[10, 10, 10], pbc=True)
    # Attach a calculator so that atoms.calc.results exists
    atoms.calc = SinglePointCalculator(  # type: ignore[no-untyped-call]
        atoms, energy=0.0, forces=[[0, 0, 0]]
    )
    return atoms
