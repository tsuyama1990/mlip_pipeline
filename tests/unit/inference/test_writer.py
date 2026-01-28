from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.inference.writer import LammpsInputWriter


@pytest.fixture
def mock_config():
    return InferenceConfig()


def test_writer_creates_files(mock_config, tmp_path):
    writer = LammpsInputWriter(mock_config, tmp_path)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    potential_path = Path("potential.yace")

    # Mock ase.io.write to avoid actual file writing issues or dependency on lammps format if not installed (though ASE usually has it)
    # But we want to verify it calls write with correct args.
    with patch("mlip_autopipec.inference.writer.write") as mock_write:
        input_file, data_file, log_file, dump_file = writer.write_inputs(atoms, potential_path)

        assert input_file == tmp_path / "in.lammps"
        assert data_file == tmp_path / "data.lammps"
        assert log_file == tmp_path / "log.lammps"
        assert dump_file == tmp_path / "dump.gamma"

        # Check that input file was written (ScriptGenerator content)
        assert input_file.exists()
        content = input_file.read_text()
        assert "# LAMMPS input script" in content

        # Check ase.io.write called with specorder
        mock_write.assert_called_once()
        args, kwargs = mock_write.call_args
        assert args[0] == data_file
        assert args[1] == atoms
        assert kwargs["format"] == "lammps-data"
        assert kwargs["specorder"] == ["H"]
