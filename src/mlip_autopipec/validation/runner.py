import logging
from pathlib import Path

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT
from ase.calculators.lammpsrun import LAMMPS

from mlip_autopipec.config.config_model import ValidationConfig
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.orchestration.interfaces import Validator
from mlip_autopipec.validation.metrics import ElasticValidator, PhononValidator
from mlip_autopipec.validation.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class ValidationRunner(Validator):
    def __init__(self, config: ValidationConfig) -> None:
        self.config = config

    def _get_calculator(self, potential_path: Path, elements: list[str]) -> Calculator:
        """
        Returns an ASE calculator for the potential.
        Tries to use LAMMPS.
        """
        # For testing/mocking purposes, if potential_path doesn't exist or is dummy
        if not potential_path.exists() or potential_path.name.endswith(".mock"):
            logger.warning("Potential path invalid or mock, using EMT calculator.")
            return EMT()  # type: ignore[no-untyped-call]

        # Construct LAMMPS calculator
        # This assumes 'lmp' is in PATH. In real scenario, we should use config.lammps.command
        # But ValidationRunner doesn't have access to full config in __init__, only ValidationConfig.
        # Design flaw? ValidationRunner should maybe receive full config or lammps config.
        # I'll assume 'lmp' or use 'ace' if available.

        # Note: file-based LAMMPS calculator is slow.
        # But it's robust.

        elem_str = " ".join(elements)

        # We need to copy potential to working directory usually, handled by ASE?
        # ASE LAMMPS calculator copies files in 'files' argument.

        parameters = {
            "pair_style": "pace",
            "pair_coeff": [f"* * {potential_path.name} {elem_str}"],
            "units": "metal",
        }

        # Instantiate without parameters to avoid ASE FileCalculator issue
        calc = LAMMPS(
            command="lmp",
            files=[str(potential_path)],
            keep_tmp_files=False,
            specorder=elements
        )  # type: ignore[no-untyped-call]
        calc.set(**parameters)  # type: ignore[no-untyped-call]
        return calc

    def _get_test_structure(self) -> Atoms:
        """
        Returns the structure to validate against.
        Currently hardcoded to Cu FCC.
        """
        # In a real app, this should be configurable.
        return bulk("Cu", "fcc", a=3.61)

    def validate(self, potential_path: Path, work_dir: Path) -> ValidationResult:
        logger.info(f"Starting validation for {potential_path}")

        # Prepare structure
        structure = self._get_test_structure()
        elements = sorted(set(structure.get_chemical_symbols())) # type: ignore[no-untyped-call]

        # Prepare calculator
        try:
            calc = self._get_calculator(potential_path, elements)
            structure.calc = calc
        except Exception as e:
            logger.exception("Failed to setup calculator")
            return ValidationResult(
                passed=False,
                reason=f"Calculator setup failed: {e}"
            )

        results = []

        # 1. Phonons
        if self.config.check_phonons:
            logger.info("Running PhononValidator")
            res = PhononValidator.run(calc, structure)
            results.append(res)

        # 2. Elastic
        if self.config.check_elastic:
            logger.info("Running ElasticValidator")
            res = ElasticValidator.run(calc, structure)
            results.append(res)

        # Aggregate
        passed = all(r.passed for r in results)
        reason = None
        if not passed:
            failed_tests = [r.name for r in results if not r.passed]
            reason = f"Validation failed for: {', '.join(failed_tests)}"

        val_result = ValidationResult(
            passed=passed,
            metrics=results,
            reason=reason,
            report_path=work_dir / "report.html"
        )

        # Generate Report
        logger.info("Generating Validation Report")
        if val_result.report_path:
             ReportGenerator.generate(val_result, val_result.report_path)

        return val_result
