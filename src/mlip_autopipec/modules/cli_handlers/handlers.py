"""
Handlers for CLI commands to ensure Single Responsibility Principle in app.py.
"""
import logging
from pathlib import Path

import typer
import yaml

from mlip_autopipec.core.services import load_config
from mlip_autopipec.generator import StructureBuilder
from mlip_autopipec.modules.training_orchestrator import TrainingManager
from mlip_autopipec.orchestration.database import DatabaseManager
from mlip_autopipec.surrogate.candidate_manager import CandidateManager
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline
from mlip_autopipec.utils.config_utils import validate_path_safety

logger = logging.getLogger(__name__)
console = typer.echo

class CLIHandler:
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
                 "max_iter": 100
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
        safe_file = validate_path_safety(file)
        load_config(safe_file)
        console("Validation Successful: Configuration is valid.")

    @staticmethod
    def generate_structures(config_file: Path, dry_run: bool) -> None:
        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)
        builder = StructureBuilder(config)
        structures = builder.build()

        console(f"Generated {len(structures)} structures.")

        if dry_run:
            console("Dry run: Not saving to database.")
            return

        with DatabaseManager(config.runtime.database_path) as db:
            cm = CandidateManager(db)
            for atoms in structures:
                metadata = atoms.info.copy()
                cm.create_candidate(atoms, metadata)

        console(f"Saved to {config.runtime.database_path}")

    @staticmethod
    def select_candidates(config_file: Path, n_samples: int | None, model_type: str | None) -> None:
        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)
        surrogate_conf = config.surrogate_config

        if n_samples is not None:
            surrogate_conf.n_samples = n_samples
        if model_type is not None:
            surrogate_conf.model_type = model_type

        with DatabaseManager(config.runtime.database_path) as db:
            pipeline = SurrogatePipeline(db, surrogate_conf)
            pipeline.run()

        console("Selection complete.")

    @staticmethod
    def train_potential(config_file: Path, prepare_only: bool) -> None:
        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)
        train_conf = config.training_config

        if not train_conf:
            console("No training configuration found in input.yaml")
            raise typer.Exit(code=1)

        work_dir = config.runtime.work_dir
        db_path = config.runtime.database_path

        with DatabaseManager(db_path) as db:
            manager = TrainingManager(db, train_conf, work_dir)

            if prepare_only:
                from mlip_autopipec.training.dataset import DatasetBuilder
                builder = DatasetBuilder(db)
                builder.export(train_conf, work_dir)
                console(f"Data preparation complete in {work_dir}")
                return

            result = manager.run_training()

            if result.success:
                console("Training successful!")
                if result.metrics:
                    console(f"Metrics: {result.metrics}")
                if result.potential_path:
                    console(f"Potential saved to: {result.potential_path}")
            else:
                console("Training failed.")
                raise typer.Exit(code=1)

    @staticmethod
    def init_db(config_file: Path) -> None:
        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)
        db_manager = DatabaseManager(config.runtime.database_path)
        with db_manager:
            pass
        console(f"Database initialized at {config.runtime.database_path}")

    @staticmethod
    def run_loop(config_file: Path) -> None:
        from mlip_autopipec.orchestration.models import OrchestratorConfig
        from mlip_autopipec.orchestration.workflow import WorkflowManager

        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)

        wf_config = config.workflow_config if config.workflow_config else OrchestratorConfig()

        manager = WorkflowManager(config, wf_config)
        manager.run()

        console("Workflow finished.")
