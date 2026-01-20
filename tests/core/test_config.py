
import pytest
from pydantic import ValidationError

from mlip_autopipec.core.config import DFTConfig, GlobalConfig


def test_dft_config_valid(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    config = DFTConfig(
        code="quantum_espresso",
        command="pw.x",
        pseudopotential_dir=pseudo_dir,
        scf_convergence_threshold=1e-8,
        mixing_beta=0.5,
        smearing="gauss"
    )
    assert config.code == "quantum_espresso"
    assert config.mixing_beta == 0.5

def test_dft_config_defaults(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    config = DFTConfig(
        code="quantum_espresso",
        command="pw.x",
        pseudopotential_dir=pseudo_dir
    )
    assert config.scf_convergence_threshold == 1e-6
    assert config.mixing_beta == 0.7
    assert config.smearing == "mv"

def test_dft_config_invalid_pseudo_dir(tmp_path):
    with pytest.raises(ValidationError):
        DFTConfig(
            code="quantum_espresso",
            command="pw.x",
            pseudopotential_dir=tmp_path / "nonexistent"
        )

def test_dft_config_invalid_beta(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    with pytest.raises(ValidationError):
        DFTConfig(
            code="quantum_espresso",
            command="pw.x",
            pseudopotential_dir=pseudo_dir,
            mixing_beta=1.5
        )

def test_global_config_valid(tmp_path):
    config = GlobalConfig(
        project_name="TestProject",
        database_path=tmp_path / "db.sqlite",
        logging_level="INFO"
    )
    assert config.project_name == "TestProject"
