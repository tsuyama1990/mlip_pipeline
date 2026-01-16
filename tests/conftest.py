"""Fixtures for the test suite."""

import pytest
import uuid
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.config.models import (
    CutoffConfig,
    DFTConfig,
    DFTInputParameters,
    Pseudopotentials,
    SystemConfig,
    TargetSystem,
)


@pytest.fixture
def sample_system_config() -> SystemConfig:
    """Provide a sample SystemConfig for a Nickel calculation."""
    dft_params = DFTInputParameters(
        pseudopotentials=Pseudopotentials({"Ni": "Ni.pbe-n-rrkjus_psl.1.0.0.UPF"}),
        cutoffs=CutoffConfig(wavefunction=60.0, density=240.0),
        k_points=(3, 3, 3),
    )
    dft_config = DFTConfig(dft_input_params=dft_params)
    target_system = TargetSystem(
        elements=["Ni"],
        composition={"Ni": 1.0},
        crystal_structure="fcc"
    )

    return SystemConfig(
        project_name="test_project",
        run_uuid=uuid.uuid4(),
        target_system=target_system,
        dft_config=dft_config,
        db_path="test.db",
        inference_config=None,
        training_config=None,
        explorer_config=None
    )


@pytest.fixture
def sample_atoms() -> Atoms:
    """Provide a sample single-atom ASE Atoms object."""
    atoms = Atoms("Ni", positions=[(0, 0, 0)], cell=[10, 10, 10], pbc=True)
    # Attach a calculator so that atoms.calc.results exists
    atoms.calc = SinglePointCalculator(  # type: ignore[no-untyped-call]
        atoms, energy=0.0, forces=[[0, 0, 0]]
    )
    return atoms
