"""Unit tests for the DFTFactory orchestrator."""

from unittest.mock import MagicMock, patch

from ase import Atoms

from mlip_autopipec.config.system import SystemConfig
from mlip_autopipec.modules.dft.factory import DFTFactory


@patch("mlip_autopipec.modules.dft.input_generator.QEInputGenerator.generate")
@patch("mlip_autopipec.modules.dft.process_runner.QEProcessRunner.execute")
@patch("mlip_autopipec.modules.dft.output_parser.QEOutputParser.parse")
def test_dftfactory_run_orchestration(
    mock_parse: MagicMock,
    mock_execute: MagicMock,
    mock_generate: MagicMock,
    sample_system_config: SystemConfig,
    sample_atoms: Atoms,
) -> None:
    """Test that the DFTFactory correctly orchestrates its components."""
    mock_generate.return_value = "dummy input file content"
    mock_parse.return_value = {"energy": -1.0, "forces": [[0, 0, 0]], "stress": [0] * 6}

    factory = DFTFactory(sample_system_config)
    result_atoms = factory.run(sample_atoms)

    mock_generate.assert_called_once_with(sample_atoms)
    mock_execute.assert_called_once()
    mock_parse.assert_called_once()
    assert result_atoms.calc.results["energy"] == -1.0
