"""Fixtures for the test suite."""

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.config_schemas import (
    DFTConfig,
    DFTExecutable,
    DFTInput,
    DFTSystem,
    SystemConfig,
    TargetSystem,
)


@pytest.fixture
def sample_system_config() -> SystemConfig:
    """Provide a sample SystemConfig for a Nickel calculation."""
    target_system = TargetSystem(elements=["Ni"], composition={"Ni": 1.0})
    dft_config = DFTConfig(
        executable=DFTExecutable(command="mock_pw.x"),
        input=DFTInput(
            pseudopotentials={"Ni": "Ni.pbe-n-rrkjus_psl.1.0.0.UPF"},
            system=DFTSystem(nat=1, ntyp=1, ecutwfc=60.0, nspin=2),
        ),
    )
    return SystemConfig(target_system=target_system, dft=dft_config, db_path="test.db")


@pytest.fixture
def sample_atoms() -> Atoms:
    """Provide a sample single-atom ASE Atoms object."""
    atoms = Atoms("Ni", positions=[(0, 0, 0)], cell=[10, 10, 10], pbc=True)
    # Attach a calculator so that atoms.calc.results exists
    atoms.calc = SinglePointCalculator(  # type: ignore[no-untyped-call]
        atoms, energy=0.0, forces=[[0, 0, 0]]
    )
    return atoms
