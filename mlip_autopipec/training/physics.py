"""
Module for physics-based calculations.
Currently implements the ZBL (Ziegler-Biersack-Littmark) potential for short-range repulsion.
"""

import logging
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

logger = logging.getLogger(__name__)


class ZBLCalculator(Calculator):
    """
    Calculates ZBL (Ziegler-Biersack-Littmark) baseline energy and forces.

    This acts as a screened Coulomb potential for short-range repulsion.
    V(r) = (Z1*Z2*e^2/r) * phi(r/a)

    The universal screening function phi(x) is approximated by a sum of 4 exponentials.
    """

    implemented_properties = ["energy", "forces"]

    # Universal screening function coefficients (phi)
    _COEFFS = [0.1818, 0.5099, 0.2802, 0.02817]
    _EXPONENTS = [3.2, 0.9423, 0.4029, 0.2016]

    # Cutoff distance to avoid singularity (Angstroms)
    _R_MIN = 1e-4

    def calculate(self, atoms: Atoms, properties=None, system_changes=all_changes):
        """
        Performs the ZBL calculation.

        Args:
            atoms: ASE Atoms object.
            properties: List of properties to calculate (default: energy, forces).
            system_changes: List of changes (positions, numbers, cell, pbc).
        """
        super().calculate(atoms, properties, system_changes)

        energy = 0.0
        forces = np.zeros((len(atoms), 3))

        # Simple N^2 loop for pairwise interactions
        # Note: In production, this should be optimized (neighbor list).
        # However, ZBL is short-ranged and typically only significant for very close pairs.

        positions = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()

        # Constants
        # e^2 in eV*Angstrom = 14.3996
        KE2 = 14.3996

        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                dist_vec = positions[i] - positions[j]
                r = np.linalg.norm(dist_vec)

                if r < self._R_MIN:
                    logger.warning(
                        f"Atoms {i} (Z={numbers[i]}) and {j} (Z={numbers[j]}) are too close (r={r:.2e} A). "
                        "Skipping ZBL interaction to avoid singularity."
                    )
                    continue

                z1 = numbers[i]
                z2 = numbers[j]

                # Screening length a
                # a = 0.8854 * a0 / (Z1^0.23 + Z2^0.23)
                # a0 (Bohr radius) = 0.529 Angstroms
                a = (0.8854 * 0.529) / (z1**0.23 + z2**0.23)

                x = r / a
                phi = sum(c * np.exp(-d * x) for c, d in zip(self._COEFFS, self._EXPONENTS))
                dphi_dx = sum(
                    -c * d * np.exp(-d * x) for c, d in zip(self._COEFFS, self._EXPONENTS)
                )

                # V = (Z1 Z2 e^2 / r) * phi
                coulomb = (z1 * z2 * KE2) / r
                v_pair = coulomb * phi

                energy += v_pair

                # Forces
                # F = -dV/dr
                # V = K/r * phi(r/a)
                # dV/dr = -K/r^2 * phi + K/r * (1/a) * dphi/dx
                #       = - (V/r) + (K/r) * (1/a) * dphi_dx

                dv_dr = -v_pair / r + (coulomb / a) * dphi_dx

                # Force on i = -grad_i V = - (dV/dr * dr/dRi)
                # dr/dRi = (Ri - Rj) / r = dist_vec / r
                # F_i = -dv_dr * (dist_vec / r)

                force_on_i = -dv_dr * (dist_vec / r)

                forces[i] += force_on_i
                forces[j] -= force_on_i

        self.results["energy"] = energy
        self.results["forces"] = forces
