import pytest
from pydantic import ValidationError
from mlip_autopipec.domain_models.dynamics import EonConfig, EonResult
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.structure import Structure
from pathlib import Path
import ase

def test_eon_config_defaults():
    config = EonConfig()
    assert config.command == "eon"
    assert config.timeout == 86400
    assert config.config_file == "config.ini"

def test_eon_config_command_validation():
    # Safe commands
    EonConfig(command="eon")
    # EonConfig(command="/usr/bin/eon") # This fails if path doesn't exist

    # Unsafe commands
    with pytest.raises(ValueError):
        EonConfig(command="eon; rm -rf /")

    with pytest.raises(ValueError):
        EonConfig(command="python script.py") # 'python' not allowed by validate_command unless safe

def test_eon_result_instantiation():
    atoms = ase.Atoms("H2", positions=[[0,0,0], [0,0,1]])
    structure = Structure.from_ase(atoms)

    result = EonResult(
        job_id="test_job",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp/test"),
        duration_seconds=10.0,
        log_content="done",
        final_structure=structure,
        max_gamma=5.5
    )
    assert result.max_gamma == 5.5
    assert result.final_structure is not None
