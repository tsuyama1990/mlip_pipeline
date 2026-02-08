import logging
from collections.abc import Iterable, Iterator

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
            # Add other defaults if needed
        }
        return Espresso(**params)  # type: ignore[no-untyped-call]

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

        Args:
            structures: An iterable of unlabeled Structure objects.

        Yields:
            Structure: Labeled Structure objects.
        """
        calc_wrapper = QECalculator(self.config)

        for structure in structures:
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
                # Log error and skip structure
                # We catch Exception broadly because calculator might raise various errors (EspressoError, PropertyNotImplementedError, etc.)
                logger.exception("Failed to compute structure")
                continue

    def _compute_single(self, atoms: Atoms, calc_wrapper: QECalculator) -> None:
        """Run calculation on single atoms object with healing."""
        # Initial calculator
        calc = calc_wrapper.create_calculator()
        atoms.calc = calc

        # Max retries can be inferred from Healer strategies or hardcoded limit
        max_retries = 5
        attempts = 0

        while attempts < max_retries:
            try:
                atoms.get_potential_energy()  # type: ignore[no-untyped-call] # Trigger calculation
            except Exception as e:
                attempts += 1
                logger.warning(f"Calculation failed (attempt {attempts}): {e}")

                if attempts >= max_retries:
                    msg = "Max retries reached"
                    raise HealingFailedError(msg) from e

                try:
                    # Heal calculator
                    # Healer expects Calculator and Exception
                    if atoms.calc is None:
                        msg = "Calculator is None"
                        raise HealingFailedError(msg)  # noqa: TRY301

                    # atoms.calc is narrowed to Calculator here
                    new_calc = self.healer.heal(atoms.calc, e)
                    atoms.calc = new_calc
                except HealingFailedError:
                    logger.exception("Healing failed")
                    raise
            else:
                # If success, store parameters in info for provenance
                if hasattr(atoms.calc, "parameters"):
                    # Copy parameters to avoid modifying calculator internal state if reused (though we create new one each time)
                    atoms.info["qe_params"] = atoms.calc.parameters.copy()
                return
