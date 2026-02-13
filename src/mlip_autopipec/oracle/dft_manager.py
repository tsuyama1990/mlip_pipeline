import logging
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed

from mlip_autopipec.domain_models.config import OracleConfig
from mlip_autopipec.domain_models.datastructures import Structure
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
        # Note: structure.to_ase() creates a copy, but we need to ensure calculator is fresh.
        # Actually, run_with_healing takes atoms and config.
        # But run_with_healing expects atoms to have a calculator attached.

        atoms = structure.ase_atoms

        # Create driver and attach calculator
        driver = QEDriver(config)
        calc = driver.get_calculator(atoms)
        atoms.calc = calc

        # Run calculation with healing
        run_with_healing(atoms, config)

        # Extract results
        energy = atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        forces = atoms.get_forces().tolist()  # type: ignore[no-untyped-call]
        stress = atoms.get_stress().tolist()  # type: ignore[no-untyped-call]

        # Update structure
        # We return a new Structure with updated properties
        # But wait, structure is immutable Pydantic?
        # We can construct a new one.
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
        """
        logger.info(f"Starting DFT calculations with {self.n_workers} workers.")

        # Use ProcessPoolExecutor for CPU-bound tasks (DFT is heavy, but usually handled by MPI outside)
        # If n_workers > 1, we run multiple DFT instances.
        # Be careful with MPI inside multiprocessing.

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Note: _process_structure_wrapper must be picklable
            future_to_structure = {
                executor.submit(_process_structure_wrapper, s, self.config): s for s in structures
            }

            for future in as_completed(future_to_structure):
                s = future_to_structure[future]
                try:
                    result = future.result()
                    # If the task returned a structure (even a failed one), yield it
                    # If process_structure raised an unhandled exception, it's caught here
                    yield result
                except Exception:
                    logger.exception(f"Structure {s} failed execution")
                    # In case of process crash, we might want to yield a placeholder?
                    # Or just log.
                    # Yielding a failed placeholder is safer for downstream to know it's done.
                    yield Structure(
                        atoms=s.atoms,
                        provenance=s.provenance,
                        label_status="failed"
                    )
