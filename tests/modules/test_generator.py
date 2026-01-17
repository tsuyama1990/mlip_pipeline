import uuid

import numpy as np
import pytest
from ase.build import bulk

from mlip_autopipec.config.models import (
    AlloyParams,
    Composition,
    CrystalParams,
    CutoffConfig,
    DFTConfig,
    DFTInputParameters,
    GeneratorParams,
    Pseudopotentials,
    SystemConfig,
    TargetSystem,
)
from mlip_autopipec.modules.generator import PhysicsInformedGenerator


@pytest.fixture
def mock_system_config() -> SystemConfig:
    """Provide a mock SystemConfig for a CuAu alloy."""
    target_system = TargetSystem(
        elements=["Cu", "Au"],
        composition=Composition({"Cu": 0.5, "Au": 0.5}),
        crystal_structure="fcc"
    )

    dft_params = DFTInputParameters(
        pseudopotentials=Pseudopotentials({"Cu": "Cu.upf", "Au": "Au.upf"}),
        cutoffs=CutoffConfig(wavefunction=30, density=120),
        k_points=(2, 2, 2)
    )

    return SystemConfig(
        project_name="TestAlloy",
        run_uuid=uuid.uuid4(),
        target_system=target_system,
        dft_config=DFTConfig(dft_input_params=dft_params),
        generator=GeneratorParams(
            alloy_params=AlloyParams(strain_magnitudes=[0.95, 1.05], rattle_std_devs=[0.1])
        ),
        inference_config=None,
        training_config=None,
        explorer_config=None
    )


@pytest.fixture
def mock_crystal_config() -> SystemConfig:
    """Provide a mock SystemConfig for a Si crystal."""
    target_system = TargetSystem(
        elements=["Si"],
        composition=Composition({"Si": 1.0}),
        crystal_structure="diamond"
    )

    dft_params = DFTInputParameters(
        pseudopotentials=Pseudopotentials({"Si": "Si.upf"}),
        cutoffs=CutoffConfig(wavefunction=30, density=120),
        k_points=(2, 2, 2)
    )

    return SystemConfig(
        project_name="TestCrystal",
        run_uuid=uuid.uuid4(),
        target_system=target_system,
        dft_config=DFTConfig(dft_input_params=dft_params),
        generator=GeneratorParams(crystal_params=CrystalParams(defect_types=["vacancy"])),
        inference_config=None,
        training_config=None,
        explorer_config=None
    )


def test_generate_alloy_workflow(mock_system_config: SystemConfig) -> None:
    """Test the full alloy generation workflow with the mock generator."""
    generator = PhysicsInformedGenerator(mock_system_config)
    generated_structures = generator.generate()

    assert len(generated_structures) == 6
    for atoms in generated_structures:
        assert atoms.get_chemical_symbols().count("Cu") == 4
        assert atoms.get_chemical_symbols().count("Au") == 4

    base_volume = generated_structures[0].get_volume()
    strained_volume = generated_structures[2].get_volume()
    assert not np.isclose(base_volume, strained_volume)

    base_pos = generated_structures[0].get_positions()
    rattled_pos = generated_structures[1].get_positions()
    assert not np.allclose(base_pos, rattled_pos)


def test_generate_crystal_workflow(mock_crystal_config: SystemConfig) -> None:
    """Test the crystal defect generation workflow with the mock generator."""
    generator = PhysicsInformedGenerator(mock_crystal_config)
    pristine_supercell = bulk("Si", "fcc", a=4.0, cubic=True).repeat((3, 3, 3))
    expected_pristine_atoms = len(pristine_supercell)

    generated_structures = generator.generate()

    assert len(generated_structures) == 1
    vacancy_structure = generated_structures[0]
    assert len(vacancy_structure) == expected_pristine_atoms - 1


def test_apply_strains() -> None:
    """Unit test for the _apply_strains method."""
    atoms = bulk("Ni", "fcc", a=3.5)
    target_system = TargetSystem(
        elements=["Ni"],
        composition=Composition({"Ni": 1.0}),
        crystal_structure="fcc"
    )

    dft_params = DFTInputParameters(
        pseudopotentials=Pseudopotentials({"Ni": "Ni.upf"}),
        cutoffs=CutoffConfig(wavefunction=30, density=120),
        k_points=(2, 2, 2)
    )

    config = SystemConfig(
        project_name="TestStrain",
        run_uuid=uuid.uuid4(),
        target_system=target_system,
        dft_config=DFTConfig(dft_input_params=dft_params),
        generator=GeneratorParams(alloy_params=AlloyParams(strain_magnitudes=[0.9, 1.1])),
        inference_config=None,
        training_config=None,
        explorer_config=None
    )
    generator = PhysicsInformedGenerator(config)

    strained_structures = generator._apply_strains(atoms)

    assert len(strained_structures) == 2
    original_volume = atoms.get_volume()
    strained_vol_1 = strained_structures[0].get_volume()
    strained_vol_2 = strained_structures[1].get_volume()
    assert np.isclose(strained_vol_1, original_volume * 0.9**3)
    assert np.isclose(strained_vol_2, original_volume * 1.1**3)


def test_apply_rattling() -> None:
    """Unit test for the _apply_rattling method."""
    atoms = bulk("Ni", "fcc", a=3.5).repeat((2, 2, 2))
    target_system = TargetSystem(
        elements=["Ni"],
        composition=Composition({"Ni": 1.0}),
        crystal_structure="fcc"
    )

    dft_params = DFTInputParameters(
        pseudopotentials=Pseudopotentials({"Ni": "Ni.upf"}),
        cutoffs=CutoffConfig(wavefunction=30, density=120),
        k_points=(2, 2, 2)
    )

    config = SystemConfig(
        project_name="TestRattle",
        run_uuid=uuid.uuid4(),
        target_system=target_system,
        dft_config=DFTConfig(dft_input_params=dft_params),
        generator=GeneratorParams(alloy_params=AlloyParams(rattle_std_devs=[0.1])),
        inference_config=None,
        training_config=None,
        explorer_config=None
    )
    generator = PhysicsInformedGenerator(config)

    rattled_structures = generator._apply_rattling(atoms)

    assert len(rattled_structures) == 1
    assert not np.allclose(
        atoms.get_positions(),
        rattled_structures[0].get_positions(),
    )
