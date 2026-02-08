from ase.calculators.calculator import Calculator

from mlip_autopipec.domain_models.config import (
    HEALER_DEGAUSS_TARGET,
    HEALER_MIXING_BETA_TARGET,
)


class HealingFailedError(Exception):
    """Raised when healing strategies are exhausted."""


class Healer:
    """
    Self-healing mechanism for DFT calculations.
    """

    def heal(self, calculator: Calculator, error: Exception) -> Calculator:
        """
        Attempts to heal a failed calculator by adjusting parameters.

        Strategies:
        1. Reduce `mixing_beta` (e.g., 0.7 -> 0.3).
        2. Increase `smearing` (electronic temperature).
        3. Change diagonalization algorithm (`david` -> `cg`).

        Args:
            calculator: The failed Calculator instance.
            error: The exception raised by the calculator.

        Returns:
            Calculator: A NEW healed Calculator instance with updated parameters.

        Raises:
            HealingFailedError: If all strategies have been tried.
        """
        # Check if calculator has parameters (most ASE calculators do)
        if not hasattr(calculator, "parameters"):
            # Cannot heal if we can't inspect/modify parameters
            msg = f"Calculator {type(calculator)} does not have parameters attribute."
            raise HealingFailedError(msg)

        # Create a COPY of parameters to avoid side effects on the input calculator
        params = calculator.parameters.copy()

        # Get current values with defaults (assuming QE defaults roughly)
        mixing_beta = params.get("mixing_beta", 0.7)
        degauss = params.get("degauss", 0.01)
        diagonalization = params.get("diagonalization", "david")

        # Strategy 1: Reduce mixing_beta
        if mixing_beta > HEALER_MIXING_BETA_TARGET + 1e-5:
            params["mixing_beta"] = HEALER_MIXING_BETA_TARGET
            # Return new instance (assuming calculator type has same constructor signature)
            return type(calculator)(**params)

        # Strategy 2: Increase degauss (smearing width)
        if degauss < HEALER_DEGAUSS_TARGET - 1e-5:
            params["degauss"] = HEALER_DEGAUSS_TARGET
            return type(calculator)(**params)

        # Strategy 3: Change diagonalization algorithm
        if diagonalization != "cg":
            params["diagonalization"] = "cg"
            return type(calculator)(**params)

        # Exhausted
        msg = f"Healing failed. All strategies exhausted for error: {error}"
        raise HealingFailedError(msg)
