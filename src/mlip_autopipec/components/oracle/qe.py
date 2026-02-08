import concurrent.futures
import contextlib
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


def _process_single_structure(structure_json: str, config: QEOracleConfig) -> str | None:
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

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:  # noqa: C901, PLR0912
        """
        Compute labels using batched parallel processing with streaming.
        Controls memory usage by limiting the number of in-flight futures and total atoms.
        """
        max_workers = self.config.max_workers
        # Use batch_size as the limit for pending tasks to control memory
        max_pending = self.config.batch_size
        max_atoms_in_flight = 100_000  # Safety limit for total atoms in memory

        iterator = iter(structures)
        futures: dict[concurrent.futures.Future[str | None], int] = {}
        pending_atoms = 0
        pending_structure: Structure | None = None

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            try:
                while True:
                    # 1. Fetch next structure if needed
                    if pending_structure is None:
                        with contextlib.suppress(StopIteration):
                            pending_structure = next(iterator)

                    # 2. Check if we can submit
                    can_submit = False
                    if pending_structure is not None:
                        n_atoms = len(pending_structure.positions)
                        # Allow submission if:
                        # - We haven't reached max tasks (max_pending)
                        # - AND (we have room in atom budget OR it's the only task)
                        if len(futures) < max_pending:
                            if pending_atoms == 0 or (
                                pending_atoms + n_atoms <= max_atoms_in_flight
                            ):
                                can_submit = True
                            elif n_atoms > max_atoms_in_flight:
                                logger.warning(
                                    f"Structure with {n_atoms} atoms exceeds safety limit "
                                    f"({max_atoms_in_flight}). Processing sequentially."
                                )
                                # If it's huge, we wait until pending_atoms == 0 (handled by 'pending_atoms == 0' check)

                    # 3. Submit task
                    if can_submit and pending_structure:
                        n_atoms = len(pending_structure.positions)
                        future = executor.submit(
                            _process_single_structure,
                            pending_structure.model_dump_json(),
                            self.config,
                        )
                        futures[future] = n_atoms
                        pending_atoms += n_atoms
                        pending_structure = None
                        # Loop back to try filling more if possible
                        continue

                    # 4. Exit condition
                    if not futures and pending_structure is None:
                        break

                    # 5. Wait for results
                    done, _ = concurrent.futures.wait(
                        futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    # Process completed futures
                    for future in done:
                        atoms_freed = futures.pop(future)
                        pending_atoms -= atoms_freed
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
