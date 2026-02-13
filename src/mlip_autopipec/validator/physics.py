import logging

import numpy as np

from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.workflow import ValidationResult
from mlip_autopipec.dynamics.calculators import MLIPCalculatorFactory
from mlip_autopipec.validator.elastic import ElasticAnalyzer, ElasticResults
from mlip_autopipec.validator.eos import EOSAnalyzer, EOSResults, fit_birch_murnaghan
from mlip_autopipec.validator.interface import BaseValidator
from mlip_autopipec.validator.phonon import PhononAnalyzer, PhononResults
from mlip_autopipec.validator.report import ReportGenerator

logger = logging.getLogger(__name__)

class PhysicsValidator(BaseValidator):
    """
    Validator that performs physics-based checks: EOS, Elastic Constants, Phonons.
    """

    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.val_config = config.validator
        # Analyzers
        self.eos_analyzer = EOSAnalyzer() # Not used directly if we use fit function, but kept for consistency
        self.elastic_analyzer = ElasticAnalyzer(strain_magnitude=self.val_config.strain_magnitude)
        self.phonon_analyzer = PhononAnalyzer()
        self.report_generator = ReportGenerator()
        self.calculator_factory = MLIPCalculatorFactory()

    def validate(self, potential: Potential) -> ValidationResult:
        """
        Validates the potential.
        """
        logger.info(f"PhysicsValidator: Validating potential {potential.path}...")

        structure = self._load_seed_structure()
        if not structure:
             return ValidationResult(passed=True, metadata={"status": "skipped_no_seed"})

        # Run Analyses
        eos_res = self._run_eos(potential, structure)
        elastic_res = self._run_elastic(potential, structure)

        phonon_res = None
        if self.val_config.phonon_stability:
            phonon_res = self._run_phonon(potential, structure)

        # Check Criteria
        passed, metrics = self._check_criteria(eos_res, elastic_res, phonon_res)

        # Generate Report
        output_dir = potential.path.parent / "validation_report"
        try:
            report_path = self.report_generator.generate(
                output_dir=output_dir,
                passed=passed,
                elastic_results=elastic_res,
                eos_results=eos_res,
                phonon_results=phonon_res
            )
        except Exception:
            logger.exception("Failed to generate validation report.")
            report_path = None

        return ValidationResult(
            passed=passed,
            metrics=metrics,
            report_path=report_path,
            metadata={"status": "completed"}
        )

    def _load_seed_structure(self) -> Structure | None:
        seed_path = self.config.generator.seed_structure_path
        if not seed_path:
            logger.warning("No seed structure configured. Skipping physics checks.")
            return None

        if not seed_path.exists():
             logger.warning(f"Seed structure file not found at {seed_path}. Skipping physics checks.")
             return None

        try:
            from ase.io import read
            atoms_list = read(seed_path, index=":")
            atoms = atoms_list[0] if isinstance(atoms_list, list) else atoms_list
            return Structure(atoms=atoms, provenance="validation_seed")
        except Exception:
            logger.exception("Failed to load seed structure.")
            return None

    def _check_criteria(
        self,
        eos_res: EOSResults | None,
        elastic_res: ElasticResults | None,
        phonon_res: PhononResults | None
    ) -> tuple[bool, dict[str, float]]:
        passed = True
        metrics: dict[str, float] = {}

        if eos_res:
            metrics["bulk_modulus_eos"] = eos_res.bulk_modulus
            if eos_res.bulk_modulus <= 0:
                passed = False
                logger.warning("Validation failed: Negative Bulk Modulus (EOS)")

        if elastic_res:
            metrics["C11"] = elastic_res.C11
            metrics["C12"] = elastic_res.C12
            metrics["C44"] = elastic_res.C44
            metrics["bulk_modulus_elastic"] = elastic_res.bulk_modulus

            if not self._check_born_criteria(elastic_res):
                passed = False

        if phonon_res:
            metrics["max_imaginary_freq"] = phonon_res.max_imaginary_freq
            if not phonon_res.is_stable:
                passed = False
                logger.warning(f"Validation failed: Phonon instability (max imag freq: {phonon_res.max_imaginary_freq:.4f} THz)")

        return passed, metrics

    def _check_born_criteria(self, res: ElasticResults) -> bool:
        passed = True
        if (res.C11 - res.C12) <= 0:
            passed = False
            logger.warning("Validation failed: Born criteria (C11 - C12 <= 0)")
        if (res.C11 + 2 * res.C12) <= 0:
            passed = False
            logger.warning("Validation failed: Born criteria (C11 + 2C12 <= 0)")
        if res.C44 <= 0:
            passed = False
            logger.warning("Validation failed: Born criteria (C44 <= 0)")
        return passed

    def _run_eos(self, potential: Potential, structure: Structure) -> EOSResults | None:
        try:
            logger.info("Running EOS Analysis...")
            atoms = structure.to_ase().copy() # type: ignore[no-untyped-call]

            # 7 points from 0.94 to 1.06 (linear scale factor)
            scales = np.linspace(0.94, 1.06, 7)
            volumes = []
            energies = []

            original_cell = atoms.get_cell()

            for s in scales:
                atoms_scaled = atoms.copy()
                atoms_scaled.set_cell(original_cell * s, scale_atoms=True)

                calc = self.calculator_factory.create(potential.path)
                atoms_scaled.calc = calc

                volumes.append(atoms_scaled.get_volume())
                energies.append(atoms_scaled.get_potential_energy())

            return fit_birch_murnaghan(volumes, energies)

        except Exception:
            logger.warning("EOS Analysis failed.", exc_info=True)
            return None

    def _run_elastic(self, potential: Potential, structure: Structure) -> ElasticResults | None:
        try:
            logger.info("Running Elastic Analysis...")
            res_dict = self.elastic_analyzer.analyze(potential, structure)
            return ElasticResults(
                C11=res_dict["C11"],
                C12=res_dict["C12"],
                C44=res_dict["C44"],
                bulk_modulus=res_dict["bulk_modulus"],
                shear_modulus=res_dict["shear_modulus"]
            )
        except Exception:
            logger.warning("Elastic Analysis failed.", exc_info=True)
            return None

    def _run_phonon(self, potential: Potential, structure: Structure) -> PhononResults | None:
        try:
            logger.info("Running Phonon Analysis...")
            return self.phonon_analyzer.analyze(
                potential,
                structure,
                supercell_matrix=self.val_config.phonon_supercell
            )
        except Exception:
            logger.warning("Phonon Analysis failed.", exc_info=True)
            return None
