"""Validator manager."""

import os
from pathlib import Path

from ase import Atoms
from loguru import logger

from pyacemaker.core.config import ValidatorConfig
from pyacemaker.domain_models.models import Potential, PotentialType
from pyacemaker.domain_models.validator import ValidationResult
from pyacemaker.validator.physics import check_elastic, check_eos, check_phonons
from pyacemaker.validator.report import ReportGenerator


class ValidatorManager:
    """Validator orchestrator."""

    def __init__(self, config: ValidatorConfig) -> None:
        """Initialize validator manager."""
        self.config = config
        self.logger = logger.bind(name="ValidatorManager")

    def _attach_calculator(self, atoms: Atoms, potential: Potential) -> Atoms:
        """Attach potential calculator to atoms."""
        if atoms.calc is not None:
            return atoms

        potential_path = Path(potential.path)

        # MACE Support
        if potential.type == PotentialType.MACE:
             try:
                 from mace.calculators import MACECalculator
                 # Simple loading for validation (CPU default for stability)
                 calc = MACECalculator(
                     model_paths=str(potential_path),
                     device="cpu",
                     default_dtype="float64"
                 )
                 atoms.calc = calc
                 self.logger.info("Attached MACE calculator.")
                 return atoms
             except ImportError as e:
                 self.logger.warning("MACE not installed.")
                 raise RuntimeError("MACE missing") from e
             except Exception as e:
                 self.logger.exception("Failed to attach MACE calculator")
                 raise RuntimeError("MACE failure") from e

        # Try attaching real PACE calculator (Default or PACE type)
        try:
            from ase.calculators.lammpslib import LAMMPSlib

            if not potential_path.exists():
                msg = f"Potential file {potential_path} not found. Cannot attach LAMMPSlib."
                self.logger.warning(msg)
                raise FileNotFoundError(msg)  # noqa: TRY301

            pot_path_str = str(potential_path.resolve())
            # Use set comprehension for unique elements
            elements = sorted({*atoms.get_chemical_symbols()})  # type: ignore[no-untyped-call]
            elem_str = " ".join(elements)

            cmds = [
                "pair_style pace",
                f"pair_coeff * * {pot_path_str} {elem_str}",
            ]

            calc = LAMMPSlib(lammps_header=cmds, log_file="lammps.log")  # type: ignore[no-untyped-call]
            atoms.calc = calc
            self.logger.info("Attached LAMMPSlib (PACE) calculator.")
            return atoms  # noqa: TRY300

        except (ImportError, RuntimeError, FileNotFoundError) as e:
            self.logger.exception("Could not attach LAMMPSlib calculator")
            msg = "Failed to attach production calculator."
            raise RuntimeError(msg) from e

    def validate(
        self, potential: Potential, structure: Atoms, output_dir: Path
    ) -> ValidationResult:
        """Run validation."""
        output_dir.mkdir(parents=True, exist_ok=True)
        potential_path = Path(potential.path)

        self.logger.info(
            f"Validating potential {potential_path} on structure {structure.get_chemical_formula()}"  # type: ignore[no-untyped-call]
        )

        # Attach calculator to structure
        try:
            structure = self._attach_calculator(structure, potential)
        except RuntimeError:
            self.logger.exception("Skipping validation due to missing calculator.")
            return ValidationResult(
                passed=False,
                metrics={},
                phonon_stable=False,
                elastic_stable=False,
                artifacts={},
            )

        # 1. Phonons
        self.logger.info("Running phonon stability check...")
        try:
            phonon_stable = check_phonons(structure, supercell=self.config.phonon_supercell)
        except Exception:
            self.logger.exception("Phonon check failed with error")
            phonon_stable = False

        # 2. EOS
        self.logger.info("Running EOS check...")
        try:
            # Run inside output_dir to capture artifacts
            original_cwd = Path.cwd()
            os.chdir(output_dir)
            try:
                bulk_modulus, eos_plot = check_eos(structure, strain=self.config.eos_strain)
            finally:
                os.chdir(original_cwd)

            # Resolve artifact path
            eos_plot_path = output_dir / eos_plot
        except Exception:
            self.logger.exception("EOS check failed with error")
            bulk_modulus = 0.0
            eos_plot_path = Path("eos_failed.png")

        # 3. Elastic
        self.logger.info("Running elastic stability check...")
        try:
            elastic_stable, elastic_constants = check_elastic(
                structure, strain=self.config.elastic_strain
            )
        except Exception:
            self.logger.exception("Elastic check failed with error")
            elastic_stable = False
            elastic_constants = {}

        # Aggregate metrics
        metrics = {
            "bulk_modulus": bulk_modulus,
        }
        metrics.update(elastic_constants)

        # Determine overall pass
        passed = phonon_stable and elastic_stable

        artifacts = {
            "eos": str(eos_plot_path),
        }

        result = ValidationResult(
            passed=passed,
            metrics=metrics,
            phonon_stable=phonon_stable,
            elastic_stable=elastic_stable,
            artifacts=artifacts,
        )

        # 4. Report
        self.logger.info("Generating report...")
        report_generator = ReportGenerator()
        report_path = output_dir / "validation_report.html"
        report_generator.generate(result, report_path)

        self.logger.info(f"Validation complete. Passed: {passed}")
        return result
