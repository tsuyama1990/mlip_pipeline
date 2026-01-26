import logging
from pathlib import Path

import typer
import yaml

from mlip_autopipec.utils.config_utils import validate_path_safety

console = typer.echo
logger = logging.getLogger(__name__)

class ProjectHandler:
    @staticmethod
    def init_project() -> None:
        input_file = Path("input.yaml")
        if input_file.exists():
            console("input.yaml already exists.")
            return

        template = {
            "target_system": {
                "name": "FeNi System",
                "elements": ["Fe", "Ni"],
                "composition": {"Fe": 0.7, "Ni": 0.3},
                "crystal_structure": "fcc",
            },
            "dft": {
                "command": "mpirun -np 4 pw.x",
                "pseudopotential_dir": "/path/to/upf",
                "ecutwfc": 40.0,
                "kspacing": 0.15,
                "nspin": 2,
            },
            "runtime": {"database_path": "mlip.db", "work_dir": "_work"},
            "training_config": {
                 "cutoff": 5.0,
                 "b_basis_size": 300,
                 "kappa": 0.5,
                 "kappa_f": 100.0,
                 "max_iter": 100,
                 "batch_size": 32
            },
            "inference_config": {
                "lammps_executable": "/path/to/lmp",
                "temperature": 1000.0,
                "steps": 10000,
                "uncertainty_threshold": 10.0
            },
            "workflow": {
                "max_generations": 5,
                "workers": 4
            }
        }

        with open(input_file, "w") as f:
            yaml.dump(template, f, sort_keys=False)

        console("Initialized new project. Please edit input.yaml.")

    @staticmethod
    def validate_config(file: Path) -> None:
        from mlip_autopipec.core.services import load_config
        safe_file = validate_path_safety(file)
        load_config(safe_file)
        console("Validation Successful: Configuration is valid.")
