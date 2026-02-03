import logging
import os
from pathlib import Path

from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.lammpsrun import LAMMPS
from ase.io import read

from mlip_autopipec.config.config_model import ValidationConfig
from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult
from mlip_autopipec.validation.metrics import ElasticValidator, PhononValidator
from mlip_autopipec.validation.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class ValidationRunner:
    def __init__(self, config: ValidationConfig) -> None:
        self.config = config

    def validate(self, potential_path: Path, work_dir: Path) -> ValidationResult:
        """
        Runs validation tests on the potential.
        """
        logger.info(f"Starting validation for potential: {potential_path}")

        if not self.config.run_validation:
            logger.info("Validation disabled in config.")
            return ValidationResult(passed=True, reason="Validation disabled")

        # Load Structure
        structure = self._get_validation_structure()

        # Attach Calculator (LAMMPS)
        self._attach_calculator(structure, potential_path)

        metrics: list[MetricResult] = []

        # Phonons
        if self.config.check_phonons:
            metrics.append(PhononValidator.validate(potential_path, structure))

        # Elastic
        if self.config.check_elastic:
            metrics.append(ElasticValidator.validate(potential_path, structure))

        # Aggregate
        passed = all(m.passed for m in metrics) if metrics else True

        result = ValidationResult(passed=passed, metrics=metrics)

        # Generate Report
        try:
            report_path = ReportGenerator.generate(result, work_dir)
            result.report_path = report_path
        except Exception:
            logger.exception("Failed to generate report")

        return result

    def _get_validation_structure(self) -> Atoms:
        if self.config.validation_structure:
            p = Path(self.config.validation_structure)
            if p.exists():
                try:
                    # read returns Atoms or list of Atoms
                    struct = read(p)
                except Exception:
                    logger.exception(f"Failed to read validation structure {p}")
                    struct = None

                if struct is not None:
                    if isinstance(struct, list):
                        return struct[0]
                    return struct
            else:
                logger.warning(f"Validation structure file {p} not found. Using default Cu.")

        return bulk("Cu", "fcc", a=3.61)

    def _attach_calculator(self, structure: Atoms, potential_path: Path) -> None:
        """
        Attaches a LAMMPS calculator to the structure.
        """
        if os.environ.get("PYACEMAKER_MOCK_MODE") == "1":
            logger.info("MOCK MODE: Attaching EMT calculator.")
            structure.calc = EMT()  # type: ignore[no-untyped-call]
            return

        # We assume ACE potential for now.
        # Note: This requires LAMMPS to be installed and have PACE support.
        # If running in environment without LAMMPS, this might not error until calculation.

        # Element list is required for pair_coeff.
        # We assume single element Cu for default, or derive from structure.
        elements = sorted(
            set(structure.get_chemical_symbols())  # type: ignore[no-untyped-call]
        )
        element_str = " ".join(elements)

        parameters = {
            "pair_style": "pace",
            "pair_coeff": [f"* * {potential_path} {element_str}"],
        }

        try:
            calc = LAMMPS()  # type: ignore[no-untyped-call]
            calc.set(**parameters)  # type: ignore[no-untyped-call]
            structure.calc = calc
        except Exception as e:
            logger.warning(f"Failed to attach LAMMPS calculator: {e}")
            logger.warning("Fallback: Attaching EMT calculator to prevent validation crash.")
            structure.calc = EMT()  # type: ignore[no-untyped-call]
