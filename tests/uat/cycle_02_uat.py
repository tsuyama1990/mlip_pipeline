"""User Acceptance Test for Cycle 02: Physics-Informed Generator."""

from unittest.mock import patch

from ase.build import bulk

from mlip_autopipec.config_schemas import (
    AlloyParams,
    DFTConfig,
    DFTInput,
    GeneratorParams,
    SystemConfig,
    TargetSystem,
)
from mlip_autopipec.modules.generator import PhysicsInformedGenerator


def main() -> None:
    """Execute the UAT scenario for Cycle 02."""
    print("--- Starting UAT for Cycle 02: Physics-Informed Generator ---")

    # Part 1: Alloy Generation Workflow
    print("\n--- Part 1: Verifying Alloy Generation Workflow ---")

    # Define a specific configuration for a predictable output
    target_system_alloy = TargetSystem(
        elements=["Cu", "Au"], composition={"Cu": 0.5, "Au": 0.5}
    )
    generator_params_alloy = GeneratorParams(
        alloy=AlloyParams(
            strain_magnitudes=[1.05],  # 1 strain level
            rattle_std_devs=[0.1],  # 1 rattle level
        )
    )
    # A minimal DFT config is needed to satisfy the schema
    dft_config = DFTConfig(input=DFTInput(pseudopotentials={"Cu": "x", "Au": "y"}))

    system_config_alloy = SystemConfig(
        target_system=target_system_alloy,
        generator=generator_params_alloy,
        dft=dft_config,
    )

    # Mock the slow SQS generation to return a simple, predictable structure
    with patch("mlip_autopipec.modules.generator.generate_sqs") as mock_generate_sqs:
        # Define the mock to return a simple 2-atom structure
        # Create a 2-atom cell to match the ["Cu", "Au"] symbols
        mock_sqs = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
        mock_sqs.set_chemical_symbols(["Cu", "Au"])
        mock_generate_sqs.return_value = mock_sqs

        generator = PhysicsInformedGenerator(config=system_config_alloy)
        generated_structures = generator.generate()

    # The logic is: 1 (base) + 1 (strained) = 2 structures.
    # Then, rattle both of them (since rattle_std > 0), creating 2 more.
    # Total = 4 structures.
    print(f"✓ Generated {len(generated_structures)} structures for the alloy case.")
    assert len(generated_structures) == 4

    # Verify composition of one of the final structures
    final_composition = generated_structures[-1].get_chemical_symbols()
    assert final_composition.count("Cu") == 1
    assert final_composition.count("Au") == 1
    print("✓ Composition of generated alloy structures is correct.")

    # Part 2: Crystal Defect Generation Workflow
    print("\n--- Part 2: Verifying Crystal Defect Generation Workflow ---")

    target_system_crystal = TargetSystem(elements=["Si"], composition={"Si": 1.0})
    dft_config_crystal = DFTConfig(input=DFTInput(pseudopotentials={"Si": "z"}))
    system_config_crystal = SystemConfig(
        target_system=target_system_crystal,
        # Use default generator params which request vacancies
        generator=GeneratorParams(),
        dft=dft_config_crystal,
    )

    # The crystal generation uses pymatgen directly and is fast, no mocking needed
    # We will patch the `bulk` call to use a predictable primitive cell
    with patch("ase.build.bulk") as mock_bulk:
        # Diamond Si primitive cell has 2 atoms
        mock_bulk.return_value = bulk("Si", "diamond", a=5.43)

        generator = PhysicsInformedGenerator(config=system_config_crystal)
        generated_defects = generator.generate()

    # The logic creates one vacancy for each atom in the primitive cell.
    # Diamond Si primitive cell has 2 atoms, so we expect 2 vacancy structures.
    print(f"✓ Generated {len(generated_defects)} defect structures.")
    assert len(generated_defects) == 2

    # Each generated structure should have one atom removed (2 - 1 = 1 atom)
    assert len(generated_defects[0]) == 1
    assert len(generated_defects[1]) == 1
    print("✓ Number of atoms in generated defect structures is correct.")

    print("\n--- UAT for Cycle 02 Completed Successfully ---")


if __name__ == "__main__":
    main()
