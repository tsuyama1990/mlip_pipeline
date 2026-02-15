"""Validator manager."""

import os
from pathlib import Path

from ase import Atoms
from loguru import logger

from pyacemaker.core.config import ValidatorConfig
from pyacemaker.domain_models.validator import ValidationResult
from pyacemaker.validator.physics import check_elastic, check_eos, check_phonons
from pyacemaker.validator.report import ReportGenerator


class ValidatorManager:
    """Validator orchestrator."""

    def __init__(self, config: ValidatorConfig) -> None:
        """Initialize validator manager."""
        self.config = config
        self.logger = logger.bind(name="ValidatorManager")

    def _attach_calculator(self, atoms: Atoms, potential_path: Path) -> Atoms:
        """Attach potential calculator to atoms."""
        if atoms.calc is not None:
            return atoms

        # Try attaching real PACE calculator
        try:
            from ase.calculators.lammpslib import LAMMPSlib

            if not potential_path.exists():
                msg = f"Potential file {potential_path} not found. Cannot attach LAMMPSlib."
                self.logger.warning(msg)
                raise FileNotFoundError(msg)

            pot_path_str = str(potential_path.resolve())
            elements = sorted(list(set(atoms.get_chemical_symbols())))
            elem_str = " ".join(elements)

            cmds = [
                "pair_style pace",
                f"pair_coeff * * {pot_path_str} {elem_str}",
            ]

            calc = LAMMPSlib(lammps_header=cmds, log_file="lammps.log")
            atoms.calc = calc
            self.logger.info("Attached LAMMPSlib (PACE) calculator.")
            return atoms

        except (ImportError, RuntimeError, FileNotFoundError) as e:
            # Only fallback if explicitly allowed via some config, OR just fail for safety.
            # Review required safety: "Must not fallback to EMT"
            self.logger.error(f"Could not attach LAMMPSlib calculator: {e}")
            raise RuntimeError("Failed to attach production calculator.") from e

    def validate(
        self, potential_path: Path, structure: Atoms, output_dir: Path
    ) -> ValidationResult:
        """Run validation."""
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"Validating potential {potential_path} on structure {structure.get_chemical_formula()}"
        )

        # Attach calculator to structure
        try:
            structure = self._attach_calculator(structure, potential_path)
        except RuntimeError:
            self.logger.error("Skipping validation due to missing calculator.")
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
        except Exception as e:
            self.logger.error(f"Phonon check failed with error: {e}")
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
        except Exception as e:
            self.logger.error(f"EOS check failed with error: {e}")
            bulk_modulus = 0.0
            eos_plot_path = Path("eos_failed.png")

        # 3. Elastic
        self.logger.info("Running elastic stability check...")
        try:
            elastic_stable, elastic_constants = check_elastic(
                structure, strain=self.config.elastic_strain
            )
        except Exception as e:
            self.logger.error(f"Elastic check failed with error: {e}")
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
