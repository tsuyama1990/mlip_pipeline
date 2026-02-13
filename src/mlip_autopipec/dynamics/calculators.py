import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT

logger = logging.getLogger(__name__)

class CalculatorFactory(ABC):
    """Abstract Factory for creating ASE calculators."""

    @abstractmethod
    def create(self, potential_path: Path) -> Calculator:
        pass

class MLIPCalculatorFactory(CalculatorFactory):
    """Factory for MLIP calculators (M3GNet, PACE, etc)."""

    def create(self, potential_path: Path) -> Calculator:
        path_str = str(potential_path)

        # 1. M3GNet
        if "m3gnet" in path_str.lower():
            return self._create_m3gnet(potential_path)

        # 2. PACE
        if potential_path.suffix == ".yace":
            return self._create_pace(potential_path)

        # 3. Fallback
        logger.warning("Unknown potential format for %s, falling back to EMT (Mock).", potential_path)
        return self._create_fallback()

    def _create_m3gnet(self, path: Path) -> Calculator:
        try:
            import m3gnet  # noqa: F401
            from m3gnet.calculators import M3GNetCalculator
            from m3gnet.models import M3GNet, Potential

            # Assuming the path points to a model or we load default
            # For this simplified implementation, we assume path is just a marker or directory
            if path.exists() and path.is_dir():
                 model = M3GNet.load(str(path))
                 potential = Potential(model=model)
            else:
                 # Default pre-trained
                 potential = Potential(M3GNet.load())

            return M3GNetCalculator(potential=potential) # type: ignore[no-any-return]
        except ImportError:
            logger.warning("m3gnet not installed. Falling back to EMT.")
            return self._create_fallback()
        except Exception:
            logger.exception("Failed to load m3gnet")
            return self._create_fallback()

    def _create_pace(self, path: Path) -> Calculator:
        try:
            from pyace import PyACECalculator
            return PyACECalculator(filename=str(path)) # type: ignore[no-any-return]
        except ImportError:
            logger.warning("pyace not installed. Falling back to EMT.")
            return self._create_fallback()

    def _create_fallback(self) -> Calculator:
        """Returns a safe fallback calculator (EMT)."""
        return EMT() # type: ignore[no-untyped-call]
