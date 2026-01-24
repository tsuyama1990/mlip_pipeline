from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from mlip_autopipec.config.models import MLIPConfig
from mlip_autopipec.config.schemas.core import TargetSystem
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.validation.runner import ValidationRunner


@pytest.fixture
def mock_config(tmp_path):
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    (pseudo_dir / "Al.UPF").touch()

    return MLIPConfig(
        target_system=TargetSystem(
            name="Test", elements=["Al"], composition={"Al": 1.0}, crystal_structure="fcc"
        ),
        dft=DFTConfig(pseudopotential_dir=pseudo_dir, command="pw.x", kspacing=0.2, ecutwfc=30.0),
    )


@patch(
    "mlip_autopipec.validation.runner.ValidationRunner._resolve_lammps_cmd",
    return_value="/usr/bin/lmp",
)
@patch("mlip_autopipec.validation.runner.LAMMPS")
@patch("mlip_autopipec.validation.runner.PhononValidator")
def test_runner_phonon(mock_phonon, mock_lammps, mock_resolve, mock_config):
    potential_path = Path("pot.yace")
    runner = ValidationRunner(mock_config, potential_path)
    atoms = Atoms("Al")

    # Run with phonon flag
    mock_phonon.return_value.validate.return_value = True

    success = runner.run(atoms, flags={"phonon": True, "elastic": False, "eos": False})

    assert success is True
    mock_phonon.assert_called_once()
    mock_lammps.assert_called_once()


@patch(
    "mlip_autopipec.validation.runner.ValidationRunner._resolve_lammps_cmd",
    return_value="/usr/bin/lmp",
)
@patch("mlip_autopipec.validation.runner.LAMMPS")
@patch("mlip_autopipec.validation.runner.ElasticityValidator")
def test_runner_elastic_failure(mock_elastic, mock_lammps, mock_resolve, mock_config):
    potential_path = Path("pot.yace")
    runner = ValidationRunner(mock_config, potential_path)
    atoms = Atoms("Al")

    mock_elastic.return_value.validate.return_value = False

    success = runner.run(atoms, flags={"phonon": False, "elastic": True, "eos": False})

    assert success is False
    mock_elastic.assert_called_once()
