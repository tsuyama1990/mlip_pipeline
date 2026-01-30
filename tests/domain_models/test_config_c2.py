
from mlip_autopipec.domain_models.config import LammpsConfig, Config


def test_lammps_config_defaults():
    config = LammpsConfig()
    assert config.command == "lmp_serial"
    assert config.timeout == 3600
    assert config.cores == 1


def test_lammps_config_custom():
    config = LammpsConfig(command="mpirun -np 4 lmp_mpi", timeout=10, cores=4)
    assert config.command == "mpirun -np 4 lmp_mpi"
    assert config.timeout == 10
    assert config.cores == 4


def test_config_integration():
    # Test that Config can handle lammps section
    conf_dict = {
        "project_name": "Test",
        "potential": {
            "elements": ["Si"],
            "cutoff": 5.0
        },
        "lammps": {
            "command": "lmp",
            "timeout": 100,
            "cores": 2
        }
    }
    config = Config(**conf_dict)
    assert config.lammps.command == "lmp"
    assert config.lammps.timeout == 100
    assert config.lammps.cores == 2
