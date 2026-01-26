import shutil
from pathlib import Path

from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
from rich.console import Console
from rich.table import Table

from mlip_autopipec.core.services import load_config
from mlip_autopipec.utils.config_utils import validate_path_safety
from mlip_autopipec.validation.runner import ValidationRunner

rich_console = Console()

class ValidationHandler:
    @staticmethod
    def run_physics_validation(config_file: Path, phonon: bool = False, elastic: bool = False, eos: bool = False) -> None:
        safe_config = validate_path_safety(config_file)
        config = load_config(safe_config)

        # Determine modules
        modules = []
        if phonon: modules.append("phonon")
        if elastic: modules.append("elastic")
        if eos: modules.append("eos")

        if not modules:
            rich_console.print("[yellow]No validation modules selected. Use --phonon, --elastic, or --eos.[/yellow]")
            return

        # Locate Potential
        potential_path = config.runtime.work_dir / "potentials" / "potential.yace"

        if not potential_path.exists():
             # Fallback to local if explicit
             local_pot = Path("potential.yace")
             if local_pot.exists():
                 potential_path = local_pot
             else:
                 rich_console.print(f"[bold red]Error:[/bold red] Could not find potential at {potential_path} or potential.yace.")
                 return

        # Locate LAMMPS
        cmd = "lmp"
        if config.inference_config and config.inference_config.lammps_executable:
             cmd = str(config.inference_config.lammps_executable)

        # Check if cmd exists
        if not shutil.which(cmd.split()[0]):
             rich_console.print(f"[bold red]Error:[/bold red] LAMMPS executable '{cmd}' not found in PATH.")
             return

        # Setup Structure
        # Use primary element bulk for validation as a baseline
        elements = config.target_system.elements
        primary = elements[0]
        try:
            atoms = bulk(primary, crystalstructure=config.target_system.crystal_structure or 'fcc')
        except Exception:
            atoms = bulk(primary) # Fallback

        # Setup Calculator
        # Using pace pair style
        calc = LAMMPS(
            command=cmd,
            specorder=elements,
            pair_style="pace",
            pair_coeff=[f"* * {potential_path.absolute()} {' '.join(elements)}"],
            keep_tmp_files=False,
            tmp_dir=config.runtime.work_dir / "tmp_validation"
        )  # type: ignore[no-untyped-call]
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
                    val_str = "Array" # Simplify large arrays
                else:
                    val_str = str(val)

                if metric.unit:
                    val_str += f" {metric.unit}"

                table.add_row(res.module, metric.name, val_str, status)

        rich_console.print(table)
