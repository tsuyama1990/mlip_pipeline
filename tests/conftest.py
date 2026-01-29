"""Shared fixtures for tests."""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.domain_models import (
    Candidate,
    CandidateStatus,
    Config,
    LoggingConfig,
    OrchestratorConfig,
    PotentialConfig,
    Structure,
)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory that is cleaned up after test."""
    return tmp_path


@pytest.fixture
def sample_ase_atoms() -> Atoms:
    """Provide a sample ASE Atoms object."""
    atoms = Atoms(
        "Si2",
        positions=[[0, 0, 0], [1.3, 1.3, 1.3]],
        cell=[[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
        pbc=[True, True, True]
    )
    atoms.info['energy'] = -10.5
    atoms.arrays['forces'] = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    atoms.info['stress'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return atoms


@pytest.fixture
def sample_structure(sample_ase_atoms: Atoms) -> Structure:
    """Provide a sample Structure object."""
    return Structure.from_ase(sample_ase_atoms)


@pytest.fixture
def sample_candidate(sample_structure: Structure) -> Candidate:
    """Provide a sample Candidate object."""
    return Candidate(
        **sample_structure.model_dump(),
        source="test_fixture",
        status=CandidateStatus.PENDING,
        priority=1.0
    )


@pytest.fixture
def sample_config(temp_dir: Path) -> Config:
    """Provide a sample Config object."""
    return Config(
        project_name="test_project",
        potential=PotentialConfig(
            elements=["Si"],
            cutoff=4.0
        ),
        orchestrator=OrchestratorConfig(
            state_file=temp_dir / "workflow_state.json"
        ),
        logging=LoggingConfig(
             file_path=temp_dir / "test.log"
        )
    )
