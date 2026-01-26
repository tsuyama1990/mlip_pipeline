import logging
from pathlib import Path

import typer

from mlip_autopipec.config.models import SystemConfig, WorkflowConfig
from mlip_autopipec.core.services import load_config
from mlip_autopipec.generator import StructureBuilder
from mlip_autopipec.modules.training_orchestrator import TrainingManager
from mlip_autopipec.orchestration.database import DatabaseManager
from mlip_autopipec.orchestration.workflow import WorkflowManager
from mlip_autopipec.surrogate.candidate_manager import CandidateManager
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline
from mlip_autopipec.utils.config_utils import validate_path_safety

logger = logging.getLogger(__name__)
console = typer.echo

class WorkflowHandler:
    @staticmethod
    def run_loop(config_file: Path) -> None:
        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)

        wf_config = config.workflow_config if config.workflow_config else WorkflowConfig()

        manager = WorkflowManager(config, workflow_config=wf_config)
        manager.run()

        console("Workflow finished.")

    @staticmethod
    def generate_structures(config_file: Path, dry_run: bool) -> None:
        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)
        sys_conf = SystemConfig(
            target_system=config.target_system, generator_config=config.generator_config
        )
        builder = StructureBuilder(sys_conf)
        structures_iter = builder.build()

        count = 0
        if dry_run:
            # Just count
            for _ in structures_iter:
                count += 1
            console(f"Dry run: Generated {count} structures. Not saving to database.")
            return

        with DatabaseManager(config.runtime.database_path) as db:
            cm = CandidateManager(db)
            # Use batch creation if possible, but builder yields one by one.
            # We can accumulate or just insert. CandidateManager.create_candidate is safe enough for singular.
            # To be robust against OOM if yield is huge, we should chunk.

            # Simple chunking wrapper
            BATCH_SIZE = 100
            batch = []

            for atoms in structures_iter:
                metadata = atoms.info.copy()
                batch.append((atoms, metadata))
                count += 1

                if len(batch) >= BATCH_SIZE:
                    cm.create_candidates(batch)
                    batch = []

            if batch:
                cm.create_candidates(batch)

        console(f"Generated and saved {count} structures to {config.runtime.database_path}")

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
