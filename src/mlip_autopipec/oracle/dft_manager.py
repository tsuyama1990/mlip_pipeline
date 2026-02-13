import logging
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from mlip_autopipec.domain_models.config import OracleConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.oracle.interface import BaseOracle
from mlip_autopipec.oracle.qe_driver import QEDriver
from mlip_autopipec.oracle.self_healing import run_with_healing

logger = logging.getLogger(__name__)


def _process_structure_wrapper(structure: Structure, config: OracleConfig) -> Structure:
    """
    Helper function to process a single structure in a separate process.
    Must be top-level for pickling in ProcessPoolExecutor.
    """
    try:
        # Convert to ASE atoms
        # Note: structure.ase_atoms returns the internal object.
        # We should use structure.to_ase() to get a safe copy with info updated,
        # or structure.ase_atoms.copy() to ensure we have a fresh object to attach calculator to.
        # structure.ase_atoms is a property returning self.atoms.
        # If we modify self.atoms in the process, it doesn't affect the parent process (because pickling copies),
        # but cleaner to work on a copy.
        atoms = structure.ase_atoms.copy() # type: ignore[no-untyped-call]

        # Create driver and attach calculator
        driver = QEDriver(config)
        calc = driver.get_calculator(atoms)
        atoms.calc = calc

        # Run calculation with healing
        run_with_healing(atoms, config)

        # Extract results
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces().tolist()
        stress = atoms.get_stress().tolist()

        # Detach calculator to prevent pickling issues and reduce size
        atoms.calc = None

        # Return new Structure with updated properties
        return Structure(
            atoms=atoms,
            provenance=structure.provenance,
            label_status="labeled",
            energy=energy,
            forces=forces,
            stress=stress,
        )

    except Exception:
        logger.exception("Failed to process structure")
        # Return structure marked as failed
        return Structure(
            atoms=structure.atoms,
            provenance=structure.provenance,
            label_status="failed",
            energy=None,
            forces=None,
            stress=None,
        )


class DFTManager(BaseOracle):
    """
    Manages DFT calculations using a pool of workers.
    """

    def __init__(self, config: OracleConfig) -> None:
        self.config = config
        self.n_workers = config.n_workers

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """
        Computes properties for a batch of structures using parallel workers.
        Processing is streamed to avoid holding all structures in memory.
        """
        logger.info(f"Starting DFT calculations with {self.n_workers} workers.")

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Create iterator from input iterable
            structures_iter = iter(structures)

            # Keep track of active futures
            active_futures = {}

            # Fill the pipeline with 2 * n_workers tasks to keep workers busy
            initial_batch_size = max(1, self.n_workers * 2)

            for _ in range(initial_batch_size):
                try:
                    s = next(structures_iter)
                    future = executor.submit(_process_structure_wrapper, s, self.config)
                    active_futures[future] = s
                except StopIteration:
                    break

            while active_futures:
                # Wait for at least one future to complete
                done_futures, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)

                for future in done_futures:
                    s = active_futures.pop(future)
                    try:
                        result = future.result()
                        yield result
                    except Exception:
                        logger.exception(f"Structure {s} failed execution")
                        # Yield a failed placeholder
                        yield Structure(
                            atoms=s.atoms, provenance=s.provenance, label_status="failed"
                        )

                    # Submit next task if available
                    try:
                        next_s = next(structures_iter)
                        new_future = executor.submit(
                            _process_structure_wrapper, next_s, self.config
                        )
                        active_futures[new_future] = next_s
                    except StopIteration:
                        pass
