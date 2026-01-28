"""
Handlers for CLI commands to ensure Single Responsibility Principle in app.py.
"""

import logging
from pathlib import Path
from uuid import uuid4

import numpy as np
import typer
import yaml
from rich.progress import track

from mlip_autopipec.core.services import load_config
from mlip_autopipec.dft.runner import QERunner
from mlip_autopipec.domain_models.dft_models import DFTResult
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
                "max_iter": 100,
                "batch_size": 32,
            },
            "inference_config": {
                "lammps_executable": "/path/to/lmp",
                "temperature": 1000.0,
                "steps": 10000,
                "uncertainty_threshold": 10.0,
            },
            "workflow": {"max_generations": 5, "workers": 4},
        }

        with open(input_file, "w") as f:
            yaml.dump(template, f, sort_keys=False)

        console("Initialized new project. Please edit input.yaml.")

    @staticmethod
    def run_physics_validation(
        config_file: Path, phonon: bool = False, elastic: bool = False, eos: bool = False
    ) -> None:
        import shutil

        from ase.build import bulk
        from ase.calculators.lammpsrun import LAMMPS
        from rich.console import Console
        from rich.table import Table

        from mlip_autopipec.validation.runner import ValidationRunner

        rich_console = Console()
        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)

        # Determine modules
        modules = []
        if phonon:
            modules.append("phonon")
        if elastic:
            modules.append("elastic")
        if eos:
            modules.append("eos")

        if not modules:
            rich_console.print(
                "[yellow]No validation modules selected. Use --phonon, --elastic, or --eos.[/yellow]"
            )
            return

        # Locate Potential
        potential_path = config.runtime.work_dir / "potentials" / "potential.yace"

        if not potential_path.exists():
            # Fallback to local if explicit
            local_pot = Path("potential.yace")
            if local_pot.exists():
                potential_path = local_pot
            else:
                rich_console.print(
                    f"[bold red]Error:[/bold red] Could not find potential at {potential_path} or potential.yace."
                )
                return

        # Locate LAMMPS
        cmd = "lmp"
        if config.inference_config and config.inference_config.lammps_executable:
            cmd = str(config.inference_config.lammps_executable)

        # Check if cmd exists
        if not shutil.which(cmd.split()[0]):
            rich_console.print(
                f"[bold red]Error:[/bold red] LAMMPS executable '{cmd}' not found in PATH."
            )
            return

        # Setup Structure
        # Use primary element bulk for validation as a baseline
        elements = config.target_system.elements
        primary = elements[0]
        try:
            atoms = bulk(primary, crystalstructure=config.target_system.crystal_structure or "fcc")
        except Exception:
            atoms = bulk(primary)  # Fallback

        # Setup Calculator
        # Using pace pair style
        calc = LAMMPS(
            command=cmd,
            specorder=elements,
            pair_style="pace",
            pair_coeff=[f"* * {potential_path.absolute()} {' '.join(elements)}"],
            keep_tmp_files=False,
            tmp_dir=config.runtime.work_dir / "tmp_validation",
        )
        atoms.calc = calc

        # Run Validation
        runner = ValidationRunner(config.validation_config)

        with rich_console.status("[bold green]Running Physics Validation..."):
            results = runner.run(atoms, modules)

        # Report Results
        table = Table(title=f"Validation Results ({primary})")
        table.add_column("Module", style="cyan")
        table.add_column("Metric", style="magenta")
        table.add_column("Value")
        table.add_column("Status", justify="center")

        for res in results:
            if res.error:
                table.add_row(res.module, "Execution", "Error", f"[red]{res.error}[/red]")
                continue

            for metric in res.metrics:
                status = "[green]PASS[/green]" if metric.passed else "[red]FAIL[/red]"
                val = metric.value
                if isinstance(val, float):
                    val_str = f"{val:.4f}"
                elif isinstance(val, list):
                    val_str = "Array"  # Simplify large arrays
                else:
                    val_str = str(val)

                if metric.unit:
                    val_str += f" {metric.unit}"

                table.add_row(res.module, metric.name, val_str, status)

        rich_console.print(table)

    @staticmethod
    def validate_config(file: Path) -> None:
        safe_file = validate_path_safety(file)
        load_config(safe_file)
        console("Validation Successful: Configuration is valid.")

    @staticmethod
    def generate_structures(config_file: Path, dry_run: bool) -> None:
        from mlip_autopipec.config.models import SystemConfig

        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)

        # Convert MLIPConfig to SystemConfig for StructureBuilder compatibility
        sys_config = SystemConfig(
            target_system=config.target_system, generator_config=config.generator_config
        )

        builder = StructureBuilder(sys_config)
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
            for atoms in structures_iter:
                metadata = atoms.info.copy()
                cm.create_candidate(atoms, metadata)
                count += 1

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

    @staticmethod
    def run_loop(config_file: Path) -> None:
        from mlip_autopipec.config.schemas.workflow import WorkflowConfig
        from mlip_autopipec.orchestration.workflow import WorkflowManager

        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)

        wf_config = config.workflow_config if config.workflow_config else WorkflowConfig()

        manager = WorkflowManager(config, workflow_config=wf_config)
        manager.run()

        console("Workflow finished.")

    @staticmethod
    def run_cycle_02(config_file: Path, mock_dft: bool = False, dry_run: bool = False) -> None:
        """
        Executes the Cycle 02 pipeline: Generation -> DFT (Oracle) -> Database -> Training.
        """

        from mlip_autopipec.config.models import SystemConfig

        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)

        work_dir = config.runtime.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        db_path = config.runtime.database_path

        # 1. Generation
        console("[bold blue]Step 1: Structure Generation[/bold blue]")
        sys_config = SystemConfig(
            target_system=config.target_system, generator_config=config.generator_config
        )
        builder = StructureBuilder(sys_config)
        structures = list(builder.build())
        console(f"Generated {len(structures)} structures.")

        if dry_run:
            console("Dry run: Skipping DFT and Training.")
            return

        # 2. Oracle (DFT) & 3. Database
        console("[bold blue]Step 2: Oracle Calculation & Database Storage[/bold blue]")

        runner = None
        if not mock_dft:
            if not config.dft:
                console("[red]Error: DFT config missing.[/red]")
                raise typer.Exit(code=1)
            runner = QERunner(config.dft, work_dir=work_dir / "dft_runs")
        else:
            console("[yellow]Running in MOCK DFT mode.[/yellow]")

        with DatabaseManager(db_path) as db:
            db.set_system_config(sys_config)

            for atoms in track(structures, description="Running DFT..."):
                uid = str(uuid4())
                atoms.info["uid"] = uid
                atoms.info["generation"] = 0

                result = None
                if mock_dft:
                    # Mock Result
                    energy = -3.5 * len(atoms) + np.random.normal(0, 0.1)
                    forces = np.random.normal(0, 0.05, size=(len(atoms), 3)).tolist()
                    stress = np.zeros((3, 3)).tolist()
                    result = DFTResult(
                        uid=uid,
                        energy=energy,
                        forces=forces,
                        stress=stress,
                        succeeded=True,
                        converged=True,
                        wall_time=0.1,
                        parameters={"mock": True},
                    )
                elif runner:
                    try:
                        result = runner.run(atoms, uid=uid)
                    except Exception as e:
                        logger.exception(f"DFT Failed for {uid}: {e}")
                        continue

                if result and result.succeeded:
                    # Save to DB
                    db.save_dft_result(atoms, result, metadata={"status": "completed", "cycle": 0})
                else:
                    console(f"[red]DFT failed for {uid}[/red]")

            # 4. Training
            console("[bold blue]Step 3: Training[/bold blue]")

            if not config.training_config:
                console("[red]Error: Training config missing.[/red]")
                raise typer.Exit(code=1)

            training_dir = work_dir / "training"
            training_dir.mkdir(parents=True, exist_ok=True)

            manager = TrainingManager(db, config.training_config, training_dir)
            train_result = manager.run_training()

            if train_result.success:
                console("[green]Training Successful![/green]")
                console(f"Potential: {train_result.potential_path}")
            else:
                console("[red]Training Failed.[/red]")
                raise typer.Exit(code=1)
