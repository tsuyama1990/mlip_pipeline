"""
Handlers for CLI commands to ensure Single Responsibility Principle in app.py.
"""

import logging
from pathlib import Path

import typer
import yaml
from ase import Atoms
from ase.io import read

from mlip_autopipec.core.services import load_config
from mlip_autopipec.dft.runner import QERunner
from mlip_autopipec.generator import StructureBuilder
from mlip_autopipec.modules.training_orchestrator import TrainingManager
from mlip_autopipec.orchestration.database import DatabaseManager
from mlip_autopipec.orchestration.workflow import WorkflowManager
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
            "validation_config": {
                "phonon": {"enabled": True},
                "elastic": {"enabled": True},
                "eos": {"enabled": True},
                "fail_on_instability": False
            },
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
        config_file: Path,
        potential: Path | None = None,
        structure: Path | None = None,
        phonon: bool = False,
        elastic: bool = False,
        eos: bool = False,
    ) -> None:
        from ase.build import bulk
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
            # If no specific flags, run all enabled in config?
            # Or assume the user wants default suite.
            # Spec says "run the validation suite".
            # If user provides flags, they select. If none, run all.
            modules = ["phonon", "elastic", "eos"]

        # Locate Potential
        potential_path = potential or config.runtime.work_dir / "potentials" / "potential.yace"

        if not potential_path.exists():
            # Fallback to local if explicit
            local_pot = Path("potential.yace")
            if local_pot.exists():
                potential_path = local_pot
            else:
                rich_console.print(
                    f"[bold red]Error:[/bold red] Could not find potential at {potential_path}."
                )
                return

        # Setup Structure
        if structure and structure.exists():
            try:
                atoms = read(structure)
                # Handle list return
                if isinstance(atoms, list):
                    atoms = atoms[0]
            except Exception as e:
                rich_console.print(f"[bold red]Error:[/bold red] Failed to read structure: {e}")
                return
        else:
            # Use primary element bulk for validation as a baseline
            elements = config.target_system.elements
            primary = elements[0]
            try:
                atoms = bulk(primary, crystalstructure=config.target_system.crystal_structure or "fcc")
            except Exception:
                atoms = bulk(primary)  # Fallback

        # Run Validation
        runner = ValidationRunner(config.validation_config, work_dir=config.runtime.work_dir / "validation")

        rich_console.print(f"[bold]Validating potential:[/bold] {potential_path}")
        rich_console.print(f"[bold]Structure:[/bold] {atoms.get_chemical_formula()}")

        with rich_console.status("[bold green]Running Physics Validation..."):
            try:
                results = runner.run(atoms, potential_path, modules)
            except Exception as e:
                rich_console.print(f"[bold red]Validation Execution Failed:[/bold red] {e}")
                import traceback
                traceback.print_exc()
                return

        # Report Results
        table = Table(title="Validation Results")
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

        # Determine global success
        failed = any(not r.passed for r in results)
        if failed and config.validation_config.fail_on_instability:
             rich_console.print("[bold red]Validation FAILED (Instability Detected)[/bold red]")
             raise typer.Exit(code=1)
        if failed:
             rich_console.print("[bold yellow]Validation Completed with Failures (Warning)[/bold yellow]")
        else:
             rich_console.print("[bold green]Validation PASSED[/bold green]")


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
            for atoms in structures_iter:
                metadata = atoms.info.copy()
                metadata["status"] = "pending"
                db.add_structure(atoms, metadata)
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
                builder.export(output_path=work_dir / "training_data.xyz")
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
        from mlip_autopipec.config.models import WorkflowConfig

        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)

        wf_config = config.workflow_config if config.workflow_config else WorkflowConfig()

        manager = WorkflowManager(config, work_dir=config.runtime.work_dir, workflow_config=wf_config)
        manager.run()

        console("Workflow finished.")

    @staticmethod
    def run_cycle_02(config_file: Path, mock_dft: bool = False, dry_run: bool = False) -> None:
        """
        Executes the Cycle 02 pipeline: Generation -> DFT (Oracle) -> Database -> Training.
        """
        from mlip_autopipec.config.models import WorkflowConfig
        from mlip_autopipec.domain_models.state import WorkflowPhase
        from mlip_autopipec.orchestration.phases.exploration import ExplorationPhase

        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)

        if not config.workflow_config:
            config.workflow_config = WorkflowConfig()

        # Limit to 1 cycle
        config.workflow_config.max_generations = 1

        manager = WorkflowManager(config=config, work_dir=config.runtime.work_dir)

        if mock_dft:
            console("[WARNING] mock_dft flag is deprecated in new architecture.")

        if dry_run:
            console("Dry Run: Executing Exploration Only.")
            manager.state.current_phase = WorkflowPhase.EXPLORATION
            ExplorationPhase(manager).execute()
            return

        manager.run()

        console("Cycle 02 Pipeline Completed.")

    @staticmethod
    def run_dft_calc(config_file: Path, structure_path: Path) -> None:
        from ase.io import read

        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)

        if not config.dft:
            console("No DFT configuration found.")
            return

        # Handle file not found for structure in tests if mocked?
        # ase.io.read raises error if file not found.
        try:
            atoms = read(structure_path)
        except Exception as e:
            console(f"Error reading structure: {e}")
            raise typer.Exit(code=1) from e

        if not atoms:
             console(f"Could not read atoms from {structure_path}")
             raise typer.Exit(code=1)

        # Handle list return from read
        if isinstance(atoms, list):
            atoms = atoms[0]

        if not isinstance(atoms, Atoms):
             console("Invalid structure content")
             raise typer.Exit(code=1)

        work_dir = config.runtime.work_dir / "single_dft"
        runner = QERunner(config.dft, work_dir)

        result = runner.run(atoms, uid=structure_path.stem)

        if result.succeeded:
            console("DFT Calculation Successful")
            console(f"Energy: {result.energy} eV")
        else:
            console("DFT Calculation Failed")
            console(f"Error: {result.error_message}")
            raise typer.Exit(code=1)
