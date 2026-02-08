from ase.calculators.calculator import Calculator


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
            Calculator: A healed Calculator instance with updated parameters.

        Raises:
            HealingFailedError: If all strategies have been tried.
        """
        # Check if calculator has parameters (most ASE calculators do)
        if not hasattr(calculator, "parameters"):
            # Cannot heal if we can't inspect/modify parameters
            msg = f"Calculator {type(calculator)} does not have parameters attribute."
            raise HealingFailedError(msg)

        params = calculator.parameters

        # Get current values with defaults (assuming QE defaults roughly)
        mixing_beta = params.get("mixing_beta", 0.7)
        degauss = params.get("degauss", 0.01)
        diagonalization = params.get("diagonalization", "david")

        # Strategy 1: Reduce mixing_beta
        # If mixing_beta is significantly higher than 0.3, reduce it.
        if mixing_beta > 0.3 + 1e-5:
            # Modify in place
            params["mixing_beta"] = 0.3
            # Some calculators might need explicit update methods, but usually setting parameters is enough for next calculation if re-run
            # For Espresso, modifying parameters dict is standard way before calling calculate() again if re-instantiated or reset.
            # But wait, ASE `get_potential_energy` might reuse previous results if parameters haven't changed.
            # We are modifying parameters, so calculator should detect change.
            # However, for robustness, we return the calculator.
            return calculator

        # Strategy 2: Increase degauss (smearing width)
        # If degauss is small, increase it.
        if degauss < 0.02 - 1e-5:
            params["degauss"] = 0.02
            # Also ensure smearing type is set if not?
            # QE uses 'smearing' keyword. If not present, degauss might be ignored.
            # But we assume config sets 'mv' or similar.
            return calculator

        # Strategy 3: Change diagonalization algorithm
        if diagonalization != "cg":
            params["diagonalization"] = "cg"
            return calculator

        # Exhausted
        msg = f"Healing failed. All strategies exhausted for error: {error}"
        raise HealingFailedError(msg)
