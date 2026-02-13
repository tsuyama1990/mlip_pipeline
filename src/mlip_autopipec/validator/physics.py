import logging
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read
from ase import Atoms
from mlip_autopipec.domain_models.config import ValidatorConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.workflow import ValidationResult
from mlip_autopipec.validator.interface import BaseValidator
from mlip_autopipec.validator.eos import EOSAnalyzer, EOSResults
from mlip_autopipec.validator.elastic import ElasticAnalyzer, ElasticResults
from mlip_autopipec.validator.phonon import PhononAnalyzer, PhononResults
from mlip_autopipec.validator.report import ReportGenerator
from mlip_autopipec.dynamics.calculators import MLIPCalculatorFactory

logger = logging.getLogger(__name__)

class PhysicsValidator(BaseValidator):
    """
    Validates potentials against physics constraints (EOS, Elastic, Phonon).
    """

    def __init__(self, config: ValidatorConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir
        self.eos_analyzer = EOSAnalyzer()
        self.elastic_analyzer = ElasticAnalyzer(delta=config.strain_magnitude)
        self.phonon_analyzer = PhononAnalyzer(supercell_matrix=config.phonon_supercell)
        self.report_generator = ReportGenerator()

    def _calculate_eos_data(self, structure: Atoms, potential: Potential) -> tuple[list[float], list[float]]:
        """
        Calculate Energy vs Volume curve.

        Returns:
            Tuple of (volumes, energies).
        """
        factory = MLIPCalculatorFactory()
        if potential.path is None:
             msg = "Potential path is not set"
             raise ValueError(msg)
        calc = factory.create(potential.path)

        atoms = structure.copy() # type: ignore[no-untyped-call]
        atoms.calc = calc

        scales = np.linspace(0.9, 1.1, 7)
        volumes = []
        energies = []

        initial_cell = atoms.get_cell() # type: ignore[no-untyped-call]

        for s in scales:
            # Scale volume by s -> scale lengths by s^(1/3)
            length_scale = s**(1/3)
            new_cell = initial_cell * length_scale
            atoms.set_cell(new_cell, scale_atoms=True)

            # Static calculation
            e = atoms.get_potential_energy()
            v = atoms.get_volume()

            volumes.append(float(v))
            energies.append(float(e))

        return volumes, energies

    def _validate_eos(self, structure: Atoms, potential: Potential, metrics: dict[str, float]) -> tuple[bool, EOSResults]:
        try:
            volumes, energies = self._calculate_eos_data(structure, potential)
            eos_res = self.eos_analyzer.fit_birch_murnaghan(volumes, energies)
            metrics["E0"] = eos_res.E0
            metrics["V0"] = eos_res.V0
            metrics["B0"] = eos_res.B0
            metrics["B0_prime"] = eos_res.B0_prime

            if eos_res.B0 <= 0:
                logger.warning("Bulk modulus is non-positive: %.2f GPa", eos_res.B0)
                return False, eos_res
        except Exception:
            logger.exception("EOS Validation failed")
            # Create dummy result for report if needed
            return False, EOSResults(E0=0.0, V0=0.0, B0=0.0, B0_prime=0.0)
        else:
            return True, eos_res

    def _validate_elastic(self, structure: Atoms, potential: Potential, metrics: dict[str, float]) -> tuple[bool, ElasticResults]:
        try:
            elastic_res = self.elastic_analyzer.calculate_elastic_constants(structure, potential)
            metrics["C11"] = elastic_res.C11
            metrics["C12"] = elastic_res.C12
            metrics["C44"] = elastic_res.C44
            metrics["Bulk_Modulus"] = elastic_res.B
            metrics["Shear_Modulus"] = elastic_res.G

            # Born stability for cubic
            if elastic_res.B <= 0 or elastic_res.G <= 0:
                logger.warning("Elastic instability detected (B=%.2f, G=%.2f)", elastic_res.B, elastic_res.G)
                return False, elastic_res
        except Exception:
            logger.exception("Elastic Validation failed")
            return False, ElasticResults(C11=0.0, C12=0.0, C44=0.0, B=0.0, G=0.0)
        else:
            return True, elastic_res

    def _validate_phonon(self, structure: Atoms, potential: Potential, metrics: dict[str, float]) -> tuple[bool, PhononResults]:
        try:
            if self.config.phonon_stability:
                phonon_res = self.phonon_analyzer.calculate_phonons(structure, potential)
                metrics["phonon_stable"] = 1.0 if phonon_res.is_stable else 0.0
                metrics["max_imaginary_freq"] = phonon_res.max_imaginary_freq

                if not phonon_res.is_stable:
                    logger.warning("Phonon instability detected (max imag freq=%.4f)", phonon_res.max_imaginary_freq)
                    return False, phonon_res
                return True, phonon_res
            else:
                metrics["phonon_stable"] = 1.0 # Skipped considered safe?
                return True, PhononResults(is_stable=True, max_imaginary_freq=0.0, band_structure_path=None)
        except Exception:
            logger.exception("Phonon Validation failed")
            return False, PhononResults(is_stable=False, max_imaginary_freq=0.0, band_structure_path=None)

    def validate(self, potential: Potential) -> ValidationResult:
        logger.info("Starting Physics Validation for %s...", potential.path)

        if self.config.structure_path is None:
             msg = "Structure path not configured in ValidatorConfig."
             raise ValueError(msg)

        # Load reference structure
        try:
             structure = read(self.config.structure_path) # type: ignore[no-untyped-call]
        except Exception as e:
             msg = f"Failed to load structure from {self.config.structure_path}: {e}"
             raise ValueError(msg) from e

        metrics: dict[str, float] = {}

        passed_eos, eos_res = self._validate_eos(structure, potential, metrics)
        passed_elastic, elastic_res = self._validate_elastic(structure, potential, metrics)
        passed_phonon, phonon_res = self._validate_phonon(structure, potential, metrics)

        passed = passed_eos and passed_elastic and passed_phonon

        # 4. Generate Report
        try:
            report_path: Path | None = self.work_dir / "validation_report.html"
            if report_path:
                self.report_generator.generate_report(eos_res, elastic_res, phonon_res, report_path)
        except Exception:
            logger.exception("Report generation failed")
            report_path = None

        return ValidationResult(
            passed=passed,
            metrics=metrics,
            report_path=report_path
        )
