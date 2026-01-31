from pathlib import Path

from ase.calculators.lammpsrun import LAMMPS

from mlip_autopipec.domain_models.config import Config, ValidationConfig
from typing import Literal

from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult
from mlip_autopipec.physics.structure_gen.generator import StructureGenFactory
from mlip_autopipec.physics.reporting.html_gen import ReportGenerator
from mlip_autopipec.physics.validation.eos import EOSValidator
from mlip_autopipec.physics.validation.elasticity import ElasticityValidator
from mlip_autopipec.physics.validation.phonon import PhononValidator
from mlip_autopipec.physics.validation.base import BaseValidator


class ValidationRunner:
    def __init__(self, output_dir: Path = Path("_work_validation")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, potential_path: Path, config: Config) -> ValidationResult:
        # 1. Setup Configuration
        val_config = config.validation or ValidationConfig()

        # 2. Prepare Structure (Equilibrium)
        # We use the structure generation config to get the reference structure
        generator = StructureGenFactory.get_generator(config.structure_gen)
        # Generate 1 structure
        structure_dm = generator.generate(config.structure_gen)
        structure = structure_dm.to_ase()

        # 3. Setup Calculator Creator
        def make_calculator():
            # Ensure absolute path for potential
            pot_abs = potential_path.resolve()
            elements = " ".join(config.potential.elements)

            # Helper to get command string
            cmd = config.lammps.command
            if isinstance(cmd, list):
                cmd = " ".join(cmd)

            # Default to pace
            pair_style = "pace"
            pair_coeff = [f"* * {pot_abs.name} {elements}"]

            if config.potential.pair_style == "hybrid/overlay":
                # Basic hybrid support
                z_inner = config.potential.zbl_inner_cutoff
                z_outer = config.potential.zbl_outer_cutoff
                pair_style = f"hybrid/overlay pace zbl {z_inner} {z_outer}"
                # Note: This might need more precise ZBL setup depending on ASE/LAMMPS version
                # For now, we assume simple PACE validation is sufficient for equilibrium properties
                # But we try to pass valid commands.
                pair_coeff = [
                    f"* * pace {pot_abs.name} {elements}",
                    "* * zbl 0.0 0.0" # Dummy ZBL? usually needs Z numbers.
                    # If we fail to setup hybrid properly, it crashes.
                    # Given 'pace' is default, we focus on that.
                ]

            return LAMMPS(
                command=cmd,
                files=[str(pot_abs)],
                keep_tmp_files=False,
                pair_style=pair_style,
                pair_coeff=pair_coeff,
                specorder=config.potential.elements
            )

        # 4. Run Validators
        metrics = []
        plots = {}
        status: Literal["PASS", "WARN", "FAIL"] = "PASS"

        # We need to catch specific exceptions like "Phonopy not installed"
        validators: list[BaseValidator] = []

        # EOS
        validators.append(
            EOSValidator(
                structure,
                make_calculator(),
                val_config,
                self.output_dir,
                potential_path.name,
            )
        )

        # Elasticity
        validators.append(
            ElasticityValidator(
                structure,
                make_calculator(),
                val_config,
                self.output_dir,
                potential_path.name,
            )
        )

        # Phonon
        try:
            validators.append(
                PhononValidator(
                    structure,
                    make_calculator(),
                    val_config,
                    self.output_dir,
                    potential_path.name,
                )
            )
        except RuntimeError:
            # Phonopy missing
            metrics.append(
                ValidationMetric(
                    name="Phonon",
                    value=0.0,
                    passed=False,
                    error_message="Phonopy not installed",
                )
            )
            status = "FAIL"  # Or WARN?

        for validator in validators:
            res = validator.validate()
            metrics.extend(res.metrics)
            plots.update(res.plots)
            if res.overall_status == "FAIL":
                status = "FAIL"
            elif res.overall_status == "WARN" and status != "FAIL":
                status = "WARN"

        # 5. Aggregate Result
        final_result = ValidationResult(
            potential_id=potential_path.name,
            metrics=metrics,
            plots=plots,
            overall_status=status,
        )

        # 6. Report
        reporter = ReportGenerator(self.output_dir)
        report_path = reporter.generate(final_result)

        # Copy to root
        root_report = Path("validation_report.html")
        root_report.write_text(report_path.read_text())

        return final_result
