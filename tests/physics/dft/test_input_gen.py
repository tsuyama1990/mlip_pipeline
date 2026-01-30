from pathlib import Path
import re

import numpy as np
import pytest

from mlip_autopipec.domain_models.calculation import DFTConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.dft.input_gen import InputGenerator


@pytest.fixture
def sample_structure():
    return Structure(
        symbols=["Si", "Si"],
        positions=np.array([[0.0, 0.0, 0.0], [1.36, 1.36, 1.36]]),
        cell=np.array([[2.7, 0, 0], [0, 2.7, 0], [0, 0, 2.7]]),
        pbc=(True, True, True),
    )


@pytest.fixture
def dft_config():
    return DFTConfig(
        command="pw.x",
        pseudopotentials={"Si": Path("Si.pbe-n-kjpaw_psl.1.0.0.UPF")},
        ecutwfc=40.0,
        kspacing=0.05,
    )


def test_calculate_kpoints():
    # Test 1: Small cell -> Many K-points
    cell = np.array([[2.7, 0, 0], [0, 2.7, 0], [0, 0, 2.7]])
    kpoints = InputGenerator.calculate_kpoints(cell, 0.05)
    assert np.all(np.array(kpoints) >= 40)

    # Test 2: Large cell -> Few K-points
    cell_large = np.eye(3) * 20.0
    kpoints_large = InputGenerator.calculate_kpoints(cell_large, 0.05)
    assert np.all(np.array(kpoints_large) == 7)

    # Test 3: Large kspacing -> 1 K-point (Gamma)
    kpoints_gamma = InputGenerator.calculate_kpoints(np.eye(3) * 10.0, 2.0)
    assert np.all(np.array(kpoints_gamma) >= 1)


def test_generate_input_string(sample_structure, dft_config):
    input_str = InputGenerator.generate_input(sample_structure, dft_config)

    assert "&CONTROL" in input_str

    # Use regex to be flexible with spaces
    assert re.search(r"tprnfor\s*=\s*\.true\.", input_str, re.IGNORECASE)
    assert re.search(r"tstress\s*=\s*\.true\.", input_str, re.IGNORECASE)

    assert "Si.pbe-n-kjpaw_psl.1.0.0.UPF" in input_str

    # ecutwfc might be formatted
    assert re.search(r"ecutwfc\s*=\s*40\.0", input_str, re.IGNORECASE)

    assert "ATOMIC_POSITIONS" in input_str
    assert "K_POINTS automatic" in input_str
    assert "calculation" in input_str.lower()
    assert "scf" in input_str.lower()


def test_generate_input_with_parameters(sample_structure, dft_config):
    # Test overriding parameters
    extra_params = {"mixing_beta": 0.3}
    input_str = InputGenerator.generate_input(sample_structure, dft_config, parameters=extra_params)
    assert re.search(r"mixing_beta\s*=\s*0\.3", input_str, re.IGNORECASE)
