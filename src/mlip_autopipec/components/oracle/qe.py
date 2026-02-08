import concurrent.futures
import logging
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import cast

from ase.calculators.espresso import Espresso

from mlip_autopipec.components.oracle.base import BaseOracle
from mlip_autopipec.components.oracle.healing import Healer, HealingFailedError
from mlip_autopipec.domain_models.config import QEOracleConfig
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class QECalculator:
    """Wrapper around ASE Espresso calculator."""

    def __init__(self, config: QEOracleConfig) -> None:
        self.config = config

    def create_calculator(self) -> Espresso:
        """Create a new Espresso calculator instance from config."""
        params = {
            "ecutwfc": self.config.ecutwfc,
            "ecutrho": self.config.ecutrho,
            "mixing_beta": self.config.mixing_beta,
            "kspacing": self.config.kspacing,
            "pseudopotentials": self.config.pseudopotentials,
            "smearing": self.config.smearing,
            "tprnfor": True,
            "tstress": True,
        }
        return cast(Espresso, Espresso(**params))


def _process_single_structure(
    structure_json: str, config: QEOracleConfig
) -> str | None:
    """
    Process a single structure in a separate process.
    """
    try:
        structure = Structure.model_validate_json(structure_json)
        atoms = structure.to_ase()

        # Create initial calculator
        calc_wrapper = QECalculator(config)
        atoms.calc = calc_wrapper.create_calculator()

        healer = Healer()
        max_attempts = 5

        for attempt in range(max_attempts):
            try:
                # Run calculation (potential energy triggers force/stress calc if requested)
                atoms.get_potential_energy()

                # Success
                if hasattr(atoms.calc, "parameters"):
                    atoms.info["qe_params"] = atoms.calc.parameters.copy()

                labeled_structure = Structure.from_ase(atoms)
                labeled_structure.validate_labeled()
                return labeled_structure.model_dump_json()

            except Exception as e:
                # Calculation failed
                if attempt == max_attempts - 1:
                    logger.warning(f"QE Calculation failed after {max_attempts} attempts: {e}")
                    return None

                try:
                    # Try to heal
                    new_calc = healer.heal(atoms.calc, e)
                    atoms.calc = new_calc
                    logger.info(f"Healed calculator parameters for attempt {attempt + 2}")
                except HealingFailedError:
                    logger.warning(f"Healing failed: {e}")
                    return None

    except Exception:
        logger.exception("Unexpected error processing structure")
        return None

    return None


class QEOracle(BaseOracle):
    """
    Quantum Espresso (QE) implementation of the Oracle component.
    """

    def __init__(self, config: QEOracleConfig) -> None:
        super().__init__(config)
        self.config: QEOracleConfig = config

    @property
    def name(self) -> str:
        return self.config.name

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """
        Compute labels using batched parallel processing.
        """
        batch_size = self.config.batch_size
        max_workers = self.config.max_workers

        iterator = iter(structures)

        # Instantiate executor once
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            while True:
                # We use list(islice) to create a batch.
                batch = list(islice(iterator, batch_size))
                if not batch:
                    break

                batch_json = [s.model_dump_json() for s in batch]

                futures = [
                    executor.submit(_process_single_structure, s_json, self.config)
                    for s_json in batch_json
                ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result_json = future.result()
                        if result_json:
                            yield Structure.model_validate_json(result_json)
                        else:
                            # Structure failed completely (all healing attempts failed)
                            # We choose not to yield it, effectively filtering it out.
                            # This is safe as long as downstream can handle fewer structures.
                            pass
                    except Exception:
                        logger.exception("Worker process failed")
