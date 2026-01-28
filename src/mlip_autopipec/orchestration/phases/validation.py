import logging

from ase import Atoms
from ase.build import bulk

from mlip_autopipec.orchestration.phases.base import BasePhase
from mlip_autopipec.validation.runner import ValidationRunner

logger = logging.getLogger(__name__)


class ValidationPhase(BasePhase):
    """
    Executes the validation phase of the active learning cycle.
    Runs physics-based checks (Phonon, Elastic, EOS) on the trained potential.
    """

    def execute(self) -> None:
        logger.info("Starting Validation Phase")

        potential_path = self.manager.state.latest_potential_path

        if not potential_path or not potential_path.exists():
            logger.warning("No potential found to validate. Skipping Validation Phase.")
            return

        # Determine validation structure
        # Priority 1: From Validation Config (reference_structure - not yet impl in schema but envisioned)
        # Priority 2: From System Config (target_system)

        atoms = self._get_validation_structure()

        # Determine output directory
        cycle = self.manager.state.cycle_index
        work_dir = self.manager.work_dir / f"validation_gen_{cycle}"

        runner = ValidationRunner(
            config=self.config.validation_config,
            work_dir=work_dir
        )

        logger.info(f"Validating potential {potential_path} against structure {atoms.get_chemical_formula()}")

        try:
            # Run all enabled modules
            results = runner.run(atoms, potential_path)

            passed = all(r.passed for r in results)
            if passed:
                logger.info("Validation Checks Passed.")
            else:
                logger.warning("Validation Checks Failed.")
                # Note: runner.run() already raises RuntimeError if fail_on_instability is True

        except Exception:
            logger.exception("Validation Phase Failed")
            raise

    def _get_validation_structure(self) -> Atoms:
        """
        Constructs a representative structure for validation.
        """
        system_config = self.config.target_system
        primary_element = system_config.elements[0]
        structure_type = system_config.crystal_structure or "fcc"

        try:
            atoms = bulk(primary_element, structure_type)
        except Exception:
            logger.warning(f"Could not build bulk for {primary_element} {structure_type}. Fallback to default bulk.")
            atoms = bulk(primary_element)

        return atoms
