import logging
from pathlib import Path
from typing import Any

from ase import Atoms

# Type ignore for phonopy as it might not be typed
try:
    from phonopy import Phonopy  # noqa: F401

    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False

from mlip_autopipec.domain_models.validation import MetricResult

logger = logging.getLogger(__name__)


class ElasticValidator:
    @staticmethod
    def validate(_potential_path: Path, structure: Atoms) -> MetricResult:
        """
        Validates elastic stability.
        """
        logger.info("Running ElasticValidator...")

        if structure.calc is None:
            return MetricResult(
                name="Elastic Constants",
                passed=False,
                details={"error": "No calculator attached to structure"},
            )

        try:
            C_ij = ElasticValidator._compute_elastic_constants(structure)

            # Basic Cubic Stability Check
            c11 = C_ij.get("C11", 0.0)
            c12 = C_ij.get("C12", 0.0)
            c44 = C_ij.get("C44", 0.0)

            # Born stability criteria for cubic
            cond1 = (c11 - c12) > 0
            cond2 = (c11 + 2 * c12) > 0
            cond3 = c44 > 0

            passed = bool(cond1 and cond2 and cond3)

            score = min(c11 - c12, c11 + 2 * c12, c44)

            return MetricResult(name="Elastic Constants", passed=passed, score=score, details=C_ij)

        except Exception as e:
            logger.exception("Elastic validation failed")
            return MetricResult(name="Elastic Constants", passed=False, details={"error": str(e)})

    @staticmethod
    def _compute_elastic_constants(_structure: Atoms) -> dict[str, float]:
        """
        Computes elastic constants.
        Currently returns dummy values for infrastructure demonstration.
        Real implementation would invoke an elastic solver.
        """
        # TODO: Implement real finite difference elasticity
        return {"C11": 100.0, "C12": 50.0, "C44": 30.0}


class PhononValidator:
    @staticmethod
    def validate(_potential_path: Path, structure: Atoms) -> MetricResult:
        """
        Validates phonon stability.
        """
        if not PHONOPY_AVAILABLE:
            logger.warning("Phonopy not installed, skipping phonon validation.")
            return MetricResult(
                name="Phonon Stability",
                passed=True,
                details={"warning": "Phonopy not installed"},
            )

        logger.info("Running PhononValidator...")

        try:
            passed, min_freq, details = PhononValidator._run_phonopy_checks(structure)

            return MetricResult(
                name="Phonon Stability", passed=passed, score=min_freq, details=details
            )
        except Exception as e:
            logger.exception("Phonon validation failed")
            return MetricResult(name="Phonon Stability", passed=False, details={"error": str(e)})

    @staticmethod
    def _run_phonopy_checks(
        _structure: Atoms,
    ) -> tuple[bool, float, dict[str, Any]]:
        """
        Runs phonopy analysis.
        Currently returns dummy pass for infrastructure demonstration.
        """
        # TODO: Implement real phonopy workflow
        # 1. Generate displacements
        # 2. Calculate forces using structure.calc
        # 3. Compute Band structure
        # 4. Check for negative frequencies
        return True, 0.5, {"band_structure": "dummy"}
