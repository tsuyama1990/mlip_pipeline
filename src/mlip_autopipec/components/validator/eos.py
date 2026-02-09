import logging
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator
from ase.eos import EquationOfState

from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class EOSCalc:
    """
    Calculates Equation of State (EOS) and checks for physical validity.
    """

    def __init__(self, strain_range: float = 0.2, n_points: int = 7) -> None:
        self.strain_range = strain_range
        self.n_points = n_points

    def calculate(
        self, structure: Structure, calculator: Calculator, workdir: Path
    ) -> float:
        """
        Compute EOS and return RMSE of the fit.

        Args:
            structure: Equilibrium structure.
            calculator: ASE calculator.
            workdir: Directory for calculations.

        Returns:
            rmse: Root Mean Square Error of the EOS fit.
        """
        workdir.mkdir(parents=True, exist_ok=True)
        atoms = structure.to_ase()
        atoms.calc = calculator

        volumes = []
        energies = []

        # Strains from 1 - strain to 1 + strain
        strains = np.linspace(1 - self.strain_range, 1 + self.strain_range, self.n_points)

        # We assume isotropic expansion/compression
        original_cell = atoms.get_cell() # type: ignore[no-untyped-call]

        for s in strains:
            # Scale cell
            atoms.set_cell(original_cell * s**(1/3), scale_atoms=True) # type: ignore[no-untyped-call]
            try:
                e = atoms.get_potential_energy() # type: ignore[no-untyped-call]
                v = atoms.get_volume() # type: ignore[no-untyped-call]
                volumes.append(v)
                energies.append(e)
            except Exception:
                logger.exception(f"EOS calculation failed at strain {s}")
                continue

        if len(volumes) < 4:
            logger.error("Not enough points for EOS fit")
            return 1000.0 # High RMSE

        # Fit EOS using ASE
        try:
            eos = EquationOfState(volumes, energies, eos="birchmurnaghan") # type: ignore[no-untyped-call]
            v0_fit, e0_fit, B_fit = eos.fit() # type: ignore[no-untyped-call]

            # Calculate RMSE
            energies_fit = [eos.func(v, v0_fit, e0_fit, B_fit) for v in volumes]
            rmse = np.sqrt(np.mean((np.array(energies) - np.array(energies_fit))**2))

            if B_fit <= 0:
                 logger.warning(f"EOS Fit gave non-positive Bulk Modulus: {B_fit}")
                 return 1000.0

            return float(rmse)

        except Exception as e:
            logger.exception("EOS fit failed")
            return 1000.0
