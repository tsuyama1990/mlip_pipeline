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
        return cast(Espresso, Espresso(**params))

    def calculate(self, atoms: Atoms) -> Atoms:
        """Perform calculation on atoms."""
        calc = self.create_calculator()
        atoms.calc = calc
        return atoms


class QEOracle(BaseOracle):
    """
    Quantum Espresso (QE) implementation of the Oracle component.

    This component is responsible for running DFT calculations using Quantum Espresso
    to label structures with energy, forces, and stress.
    Uses Batched processing (Specification 3.2 "Static Calculation") and Self-Healing.
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
        calc_wrapper = QECalculator(self.config)
        batch_size = self.config.batch_size

        # Create iterator from input
        iterator = iter(structures)

        while True:
            # Create a batch of structures
            # islice(iterator, batch_size) takes next batch_size elements
            batch = list(islice(iterator, batch_size))
            if not batch:
                break

            # Process batch serially (for now, safe)
            for structure in batch:
                try:
                    # Convert to ASE
                    atoms = structure.to_ase()

                    # Perform calculation with retries/healing
                    self._compute_single(atoms, calc_wrapper)

                    # Convert back to Structure (provenance stored in tags)
                    labeled_structure = Structure.from_ase(atoms)

                    # Ensure labels are present
                    labeled_structure.validate_labeled()

                    yield labeled_structure

                except (HealingFailedError, Exception):
                    # Log error and skip structure (Robustness)
                    logger.exception("Failed to compute structure")
                    continue

    def _compute_single(self, atoms: Atoms, calc_wrapper: QECalculator) -> None:
        """Run calculation on single atoms object with healing."""
        # Initial calculator
        calc = calc_wrapper.create_calculator()
        atoms.calc = calc

        max_retries = 5
        attempts = 0

        while attempts < max_retries:
            try:
                # Trigger calculation
                # get_potential_energy ensures SCF runs
                atoms.get_potential_energy()
            except Exception as e:
                attempts += 1
                logger.warning(f"Calculation failed (attempt {attempts}): {e}")

                if attempts >= max_retries:
                    msg = "Max retries reached"
                    raise HealingFailedError(msg) from e

                try:
                    # Heal calculator
                    if atoms.calc is None:
                        msg = "Calculator is None"
                        raise HealingFailedError(msg)  # noqa: TRY301

                    # Healer.heal returns a NEW calculator instance
                    new_calc = self.healer.heal(atoms.calc, e)
                    atoms.calc = new_calc
                except HealingFailedError:
                    logger.exception("Healing failed")
                    raise
            else:
                # Success
                # Store parameters in info for provenance (Specification requirement)
                if hasattr(atoms.calc, "parameters"):
                    atoms.info["qe_params"] = atoms.calc.parameters.copy()
                return
