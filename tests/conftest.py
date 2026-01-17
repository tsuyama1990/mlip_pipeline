"""Fixtures for the test suite."""

import uuid
from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.config.models import (
    Composition,
    CutoffConfig,
    DFTConfig,
    DFTInputParameters,
    MinimalConfig,
    Pseudopotentials,
    Resources,
    SystemConfig,
    TargetSystem,
)


@pytest.fixture
def sample_system_config(tmp_path: Path) -> SystemConfig:
    """Provide a sample SystemConfig for a Nickel calculation."""
    # We need to conform to SystemConfig's required fields: minimal, working_dir, db_path, log_path

    minimal = MinimalConfig(
        project_name="test_project",
        target_system=TargetSystem(
            elements=["Ni"],
            composition=Composition({"Ni": 1.0})
        ),
        resources=Resources(
            dft_code="quantum_espresso",
            parallel_cores=1
        )
    )

    dft_params = DFTInputParameters(
        pseudopotentials=Pseudopotentials({"Ni": "Ni.pbe-n-rrkjus_psl.1.0.0.UPF"}),
        cutoffs=CutoffConfig(wavefunction=60.0, density=240.0),
        k_points=(3, 3, 3),
    )
    dft_config = DFTConfig(dft_input_params=dft_params)

    return SystemConfig(
        minimal=minimal,
        working_dir=tmp_path / "work",
        db_path=tmp_path / "work" / "test.db",
        log_path=tmp_path / "work" / "log.log",
        project_name="test_project", # Optional future
        run_uuid=uuid.uuid4(),       # Optional future
        target_system=minimal.target_system, # Optional future
        dft_config=dft_config,       # Optional future
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
