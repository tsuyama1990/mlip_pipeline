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

        # Generate volumes
        v0 = atoms.get_volume()
        volumes = []
        energies = []

        # Strains from 1 - strain to 1 + strain
        strains = np.linspace(1 - self.strain_range, 1 + self.strain_range, self.n_points)

        # We assume isotropic expansion/compression
        original_cell = atoms.get_cell()

        for s in strains:
            # Scale cell
            atoms.set_cell(original_cell * s**(1/3), scale_atoms=True)
            try:
                e = atoms.get_potential_energy()
                volumes.append(atoms.get_volume())
                energies.append(e)
            except Exception as e:
                logger.warning(f"EOS calculation failed at strain {s}: {e}")
                # If fail, skip point
                continue

        if len(volumes) < 4:
            logger.error("Not enough points for EOS fit")
            return 1000.0 # High RMSE

        # Fit EOS using ASE
        try:
            eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
            v0_fit, e0_fit, B_fit = eos.fit()

            # Calculate RMSE
            # eos.func(v) returns energy
            energies_fit = [eos.func(v, v0_fit, e0_fit, B_fit) for v in volumes]
            rmse = np.sqrt(np.mean((np.array(energies) - np.array(energies_fit))**2))

            # Also check if B (Bulk Modulus) is positive
            # B is in eV/A^3. 1 eV/A^3 = 160.2 GPa.
            if B_fit <= 0:
                 logger.warning(f"EOS Fit gave non-positive Bulk Modulus: {B_fit}")
                 return 1000.0

            return rmse

        except Exception as e:
            logger.error(f"EOS fit failed: {e}")
            return 1000.0
