import concurrent.futures
import logging
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import cast

from ase import Atoms
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
        # Convert config to dict
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
        # We cast because ASE Espresso constructor is not typed
        return cast(Espresso, Espresso(**params))  # type: ignore[no-untyped-call]

    def calculate(self, atoms: Atoms) -> Atoms:
        """Perform calculation on atoms."""
        calc = self.create_calculator()
        atoms.calc = calc
        return atoms


def _process_single_structure(
    structure_json: str, config: QEOracleConfig
) -> str | None:
    """
    Process a single structure in a separate process.
    """
    try:
        # Reconstruct inputs
        structure = Structure.model_validate_json(structure_json)
        calc_wrapper = QECalculator(config)
        healer = Healer()

        atoms = structure.to_ase()

        # Calculation Logic (Self-Contained for Pickling)
        # Initial calculator
        calc = calc_wrapper.create_calculator()
        atoms.calc = calc

        max_retries = 5
        attempts = 0

        while attempts < max_retries:
            try:
                # Trigger calculation
                # get_potential_energy ensures SCF runs
                atoms.get_potential_energy()  # type: ignore[no-untyped-call]

                # Success
                if hasattr(atoms.calc, "parameters"):
                    atoms.info["qe_params"] = atoms.calc.parameters.copy()

                # Convert back to Structure
                labeled_structure = Structure.from_ase(atoms)
                labeled_structure.validate_labeled()
                return labeled_structure.model_dump_json()

            except Exception as e:
                attempts += 1

                if attempts >= max_retries:
                    return None

                try:
                    if atoms.calc is None:
                        return None

                    new_calc = healer.heal(atoms.calc, e)
                    atoms.calc = new_calc
                except HealingFailedError:
                    return None

    except Exception:
        return None

    return None


class QEOracle(BaseOracle):
    """
    Quantum Espresso (QE) implementation of the Oracle component.

    This component is responsible for running DFT calculations using Quantum Espresso
    to label structures with energy, forces, and stress.
    Uses Batched processing and ProcessPoolExecutor for parallel execution.
    """

    def __init__(self, config: QEOracleConfig) -> None:
        super().__init__(config)
        self.config: QEOracleConfig = config
        self.healer = Healer()

    @property
    def name(self) -> str:
        return self.config.name

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """
        Compute labels (energy, forces, stress) for the given structures using QE.
        Uses batched processing for scalability.

        Args:
            structures: An iterable of unlabeled Structure objects.

        Yields:
            Structure: Labeled Structure objects.
        """
        batch_size = self.config.batch_size
        max_workers = self.config.max_workers

        iterator = iter(structures)

        # Instantiate ProcessPoolExecutor once outside the loop
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            while True:
                # IMPORTANT: islice creates an iterator, but we consume it into a list
                # to create a batch. This holds 'batch_size' items in memory.
                # Since batch_size is configurable and small (default 10), this is memory safe.
                # Using list(islice(...)) is standard for batching iterators in Python.
                batch = list(islice(iterator, batch_size))
                if not batch:
                    break

                # Parallel Processing
                # We serialize to JSON to ensure clean passing between processes
                batch_json = [s.model_dump_json() for s in batch]

                # Submit tasks for the current batch
                # We do not use map because we want to handle individual failures gracefully
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
                            logger.warning("A structure failed to compute in worker process.")
                    except Exception:
                        logger.exception("Worker process failed")
