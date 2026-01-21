import pytest
from pydantic import ValidationError
from pathlib import Path
from mlip_autopipec.core.config import DFTConfig, GlobalConfig

def test_dft_config_defaults(tmp_path: Path) -> None:
    # Create a dummy pseudo dir
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()

    config = DFTConfig(
        command="pw.x",
        pseudopotential_dir=pseudo_dir
    )

    assert config.code == "quantum_espresso"
    assert config.scf_convergence_threshold == 1e-6
    assert config.mixing_beta == 0.7
    assert config.smearing == "mv"
    assert config.kpoints_density == 0.15

def test_dft_config_validation(tmp_path: Path) -> None:
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()

    # Test invalid convergence threshold
    with pytest.raises(ValidationError):
        DFTConfig(
            command="pw.x",
            pseudopotential_dir=pseudo_dir,
            scf_convergence_threshold=0
        )

    # Test invalid mixing beta
    with pytest.raises(ValidationError):
        DFTConfig(
            command="pw.x",
            pseudopotential_dir=pseudo_dir,
            mixing_beta=1.5
        )

    # Test missing pseudo dir
    with pytest.raises(ValidationError):
        DFTConfig(
            command="pw.x",
            pseudopotential_dir=tmp_path / "non_existent"
        )

def test_global_config_validation() -> None:
    # Valid config
    config = GlobalConfig(
        project_name="TestProject",
        database_path=Path("test.db"),
        logging_level="DEBUG"
    )
    assert config.project_name == "TestProject"

    # Invalid logging level
    with pytest.raises(ValidationError):
        GlobalConfig(
            project_name="TestProject",
            database_path=Path("test.db"),
            logging_level="INVALID"
        )
