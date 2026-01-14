"""Unit tests for the QEInputGenerator."""

from ase import Atoms

from mlip_autopipec.config.system import SystemConfig
from mlip_autopipec.modules.dft.input_generator import QEInputGenerator


def test_qeinputgenerator_generate(
    sample_system_config: SystemConfig, sample_atoms: Atoms
) -> None:
    """Test the generation of the QE input file content."""
    generator = QEInputGenerator(sample_system_config)
    input_content = generator.generate(sample_atoms)

    assert "&CONTROL" in input_content
    assert "calculation = 'scf'" in input_content
    assert "&SYSTEM" in input_content
    assert "nat = 1" in input_content
    assert "nspin = 2" in input_content
    assert "ATOMIC_SPECIES" in input_content
    assert "Ni 1.0 Ni.pbe-n-rrkjus_psl.1.0.0.UPF" in input_content
    assert "ATOMIC_POSITIONS {angstrom}" in input_content
    assert "Ni 0.0 0.0 0.0" in input_content
