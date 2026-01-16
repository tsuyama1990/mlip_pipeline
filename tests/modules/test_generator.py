import numpy as np
import pytest
import uuid
from ase.build import bulk

from mlip_autopipec.config.models import (
    AlloyParams,
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
        composition={"Cu": 0.5, "Au": 0.5},
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
        # Fill required fields with dummies or None if allowed
        inference_config=None,
        training_config=None,
        explorer_config=None
    )


@pytest.fixture
def mock_crystal_config() -> SystemConfig:
    """Provide a mock SystemConfig for a Si crystal."""
    target_system = TargetSystem(
        elements=["Si"],
        composition={"Si": 1.0},
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
    # Arrange
    generator = PhysicsInformedGenerator(mock_system_config)

    # Act
    generated_structures = generator.generate()

    # Assert
    assert len(generated_structures) == 6
    for atoms in generated_structures:
        assert atoms.get_chemical_symbols().count("Cu") == 4  # type: ignore[no-untyped-call]
        assert atoms.get_chemical_symbols().count("Au") == 4  # type: ignore[no-untyped-call]

    # Check for strain (compare base structure [0] with strained structure [2])
    base_volume = generated_structures[0].get_volume()  # type: ignore[no-untyped-call]
    strained_volume = generated_structures[2].get_volume()  # type: ignore[no-untyped-call]
    assert not np.isclose(base_volume, strained_volume)

    # Check for rattle (compare base structure [0] with rattled base structure [1])
    base_pos = generated_structures[0].get_positions()  # type: ignore[no-untyped-call]
    rattled_pos = generated_structures[1].get_positions()  # type: ignore[no-untyped-call]
    assert not np.allclose(base_pos, rattled_pos)


def test_generate_crystal_workflow(mock_crystal_config: SystemConfig) -> None:
    """Test the crystal defect generation workflow with the mock generator."""
    # Arrange
    generator = PhysicsInformedGenerator(mock_crystal_config)
    pristine_supercell = bulk("Si", "fcc", a=4.0, cubic=True).repeat((3, 3, 3))  # type: ignore[no-untyped-call]
    expected_pristine_atoms = len(pristine_supercell)

    # Act
    generated_structures = generator.generate()

    # Assert
    assert len(generated_structures) == 1
    vacancy_structure = generated_structures[0]
    assert len(vacancy_structure) == expected_pristine_atoms - 1


def test_apply_strains() -> None:
    """Unit test for the _apply_strains method."""
    # Arrange
    atoms = bulk("Ni", "fcc", a=3.5)
    target_system = TargetSystem(elements=["Ni"], composition={"Ni": 1.0}, crystal_structure="fcc")

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

    # Act
    strained_structures = generator._apply_strains(atoms)

    # Assert
    assert len(strained_structures) == 2
    original_volume = atoms.get_volume()  # type: ignore[no-untyped-call]
    strained_vol_1 = strained_structures[0].get_volume()  # type: ignore[no-untyped-call]
    strained_vol_2 = strained_structures[1].get_volume()  # type: ignore[no-untyped-call]
    assert np.isclose(strained_vol_1, original_volume * 0.9**3)
    assert np.isclose(strained_vol_2, original_volume * 1.1**3)


def test_apply_rattling() -> None:
    """Unit test for the _apply_rattling method."""
    # Arrange
    atoms = bulk("Ni", "fcc", a=3.5).repeat((2, 2, 2))  # type: ignore[no-untyped-call]
    target_system = TargetSystem(elements=["Ni"], composition={"Ni": 1.0}, crystal_structure="fcc")

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

    # Act
    rattled_structures = generator._apply_rattling(atoms)

    # Assert
    assert len(rattled_structures) == 1
    assert not np.allclose(
        atoms.get_positions(),
        rattled_structures[0].get_positions(),  # type: ignore[no-untyped-call]
    )
