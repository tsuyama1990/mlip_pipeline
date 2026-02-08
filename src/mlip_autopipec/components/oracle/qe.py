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
        structure = Structure.model_validate_json(structure_json)
        calc_wrapper = QECalculator(config)
        healer = Healer()

        atoms = structure.to_ase()
        calc = calc_wrapper.create_calculator()
        atoms.calc = calc

        max_retries = 5
        attempts = 0

        while attempts < max_retries:
            try:
                atoms.get_potential_energy()  # type: ignore[no-untyped-call]

                if hasattr(atoms.calc, "parameters"):
                    atoms.info["qe_params"] = atoms.calc.parameters.copy()

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
        Compute labels using batched parallel processing.
        """
        batch_size = self.config.batch_size
        max_workers = self.config.max_workers

        iterator = iter(structures)

        # Instantiate executor once
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            while True:
                # We use list(islice) to create a batch.
                # This is safe because batch_size is strictly limited (e.g. 1000) via config validation.
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
                            logger.warning("A structure failed to compute in worker process.")
                    except Exception:
                        logger.exception("Worker process failed")
