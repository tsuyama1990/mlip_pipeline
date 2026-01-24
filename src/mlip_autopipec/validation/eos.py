import logging
from typing import Any

import numpy as np
from ase import Atoms
from ase.eos import EquationOfState
from ase.units import GPa

from mlip_autopipec.config.schemas.validation import EOSConfig

logger = logging.getLogger(__name__)


class EOSValidator:
    """
    Validates the Equation of State (EOS) and checks Bulk Modulus.
    """

    def __init__(self, config: EOSConfig) -> None:
        self.config = config

    def validate(self, atoms: Atoms, calculator: Any) -> bool:
        """
        Runs EOS calculation.

        Args:
            atoms: Structure.
            calculator: ASE calculator.

        Returns:
            bool: True if Bulk Modulus > 0.
        """
        if not isinstance(atoms, Atoms):
            raise TypeError(f"Expected ase.Atoms object, got {type(atoms)}")

        volumes = []
        energies = []

        # Apply linear scaling factors
        # 1 + strain
        strains = np.linspace(
            -self.config.strain_max, self.config.strain_max, self.config.num_points
        )

        original_cell = atoms.get_cell()

        for s in strains:
            # Isotropic expansion/compression
            scale = 1.0 + s
            new_cell = original_cell * scale

            deformed = atoms.copy()  # type: ignore[no-untyped-call]
            deformed.set_cell(new_cell, scale_atoms=True)
            deformed.calc = calculator

            volumes.append(deformed.get_volume())
            try:
                e = deformed.get_potential_energy()
                energies.append(e)
            except Exception as e:
                logger.error(f"Energy calculation failed for EOS point s={s}: {e}")
                return False

        try:
            # Fit Birch-Murnaghan
            eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
            v0, e0, B = eos.fit()

            # Convert Bulk Modulus to GPa
            # B from ASE is in eV/A^3
            B_GPa = B / GPa

            logger.info(
                f"EOS Fit (Birch-Murnaghan): V0={v0:.2f} A^3, E0={e0:.4f} eV, B={B_GPa:.2f} GPa"
            )

            # Check physical validity
            if B_GPa > 0:
                return True
            logger.warning(f"Unstable Bulk Modulus detected: {B_GPa:.2f} GPa")
            return False

        except Exception as e:
            logger.error(f"EOS fitting failed: {e}")
            return False
