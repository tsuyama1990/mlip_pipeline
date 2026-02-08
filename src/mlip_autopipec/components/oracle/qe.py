import concurrent.futures
import logging
from collections.abc import Iterable, Iterator

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
        return Espresso(**params)  # type: ignore[no-untyped-call]


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
                atoms.get_potential_energy()  # type: ignore[no-untyped-call]

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
        Compute labels using batched parallel processing with streaming.
        Controls memory usage by limiting the number of in-flight futures.
        """
        max_workers = self.config.max_workers
        # Use batch_size as the limit for pending tasks to control memory
        max_pending = self.config.batch_size

        iterator = iter(structures)
        futures: set[concurrent.futures.Future[str | None]] = set()

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            try:
                while True:
                    # Fill the pool up to max_pending
                    while len(futures) < max_pending:
                        try:
                            structure = next(iterator)
                            # Submit new task
                            future = executor.submit(
                                _process_single_structure,
                                structure.model_dump_json(),
                                self.config
                            )
                            futures.add(future)
                        except StopIteration:
                            break

                    if not futures:
                        break

                    # Wait for at least one future to complete
                    # return_when=FIRST_COMPLETED ensures we yield results as soon as possible
                    # and free up slots in 'futures' set for the filling loop above.
                    done, _ = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    # Process completed futures
                    for future in done:
                        futures.remove(future)
                        try:
                            result_json = future.result()
                            if result_json:
                                yield Structure.model_validate_json(result_json)
                            else:
                                # Structure failed (logging handled in worker)
                                pass
                        except Exception:
                            logger.exception("Worker process failed unexpectedly")
            finally:
                # Cancel any pending futures if generator is closed early
                for f in futures:
                    f.cancel()
