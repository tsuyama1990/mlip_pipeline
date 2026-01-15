"""Unit tests for the QEInputGenerator."""

from ase import Atoms

from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.modules.dft.input_generator import QEInputGenerator


def test_qeinputgenerator_generate(
    sample_system_config: SystemConfig, sample_atoms: Atoms
) -> None:
    """Test the generation of the QE input file content."""
    generator = QEInputGenerator()
    input_content = generator.generate(sample_atoms, config=sample_system_config.dft)

    assert "&CONTROL" in input_content
    assert "calculation = 'scf'" in input_content
    assert "&SYSTEM" in input_content
    assert "nat = 1" in input_content
    assert "nspin = 2" in input_content
    assert "ATOMIC_SPECIES" in input_content
    assert "Ni 1.0 Ni.pbe-n-rrkjus_psl.1.0.0.UPF" in input_content
    assert "ATOMIC_POSITIONS {angstrom}" in input_content
    assert "Ni 0.0 0.0 0.0" in input_content


def test_qeinputgenerator_empty_atoms(sample_system_config: SystemConfig) -> None:
    """Test that the generator handles an empty Atoms object."""
    generator = QEInputGenerator()
    empty_atoms = Atoms()
    input_content = generator.generate(empty_atoms, config=sample_system_config.dft)
    assert "nat = 0" in input_content
    assert "ATOMIC_POSITIONS {angstrom}" in input_content
    # Check that there are no lines defining atomic positions
    assert "Ni 0.0 0.0 0.0" not in input_content
