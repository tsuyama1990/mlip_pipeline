"""
Physics-based logic for training and data processing (e.g. ZBL).
"""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


class ZBLCalculator(Calculator):
    """
    ASE Calculator implementing the Ziegler-Biersack-Littmark (ZBL) repulsive potential.
    Used for short-range repulsion in training data augmentation.
    """

    implemented_properties = ["energy", "forces"]

    # Universal screening function coefficients (phi)
    _COEFFS = [0.1818, 0.5099, 0.2802, 0.02817]
    _EXPONENTS = [3.2, 0.9423, 0.4029, 0.2016]

    # Cutoff distance to avoid singularity (Angstroms)
    # ZBL is usually short range (< 2.0 A)
    _CUTOFF = 6.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate(self, atoms: Atoms | None = None, properties=None, system_changes=None): # type: ignore[override]
        super().calculate(atoms, properties, system_changes)

        if self.atoms is None:
            msg = "Calculator has no atoms attached."
            raise ValueError(msg)

        positions = self.atoms.get_positions()
        numbers = self.atoms.get_atomic_numbers()
        n_atoms = len(self.atoms)

        energy = 0.0
        forces = np.zeros((n_atoms, 3))

        # Naive N^2 loop - suitable only for small clusters/validation
        # For production, use optimized C/Fortran implementation (like LAMMPS pair_zbl)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist_vec = positions[i] - positions[j]
                r_val = float(np.linalg.norm(dist_vec))

                # Use max to avoid singularity
                r = max(r_val, 1e-3)

                if r > self._CUTOFF:
                    continue

                z1 = numbers[i]
                z2 = numbers[j]

                # Screening length a
                # a = 0.8854 * a0 / (Z1^0.23 + Z2^0.23)
                # a0 (Bohr radius) = 0.529 Angstroms
                a = (0.8854 * 0.529) / (z1**0.23 + z2**0.23)

                x = r / a
                phi = sum(c * np.exp(-d * x) for c, d in zip(self._COEFFS, self._EXPONENTS, strict=False))
                dphi_dx = sum(
                    -c * d * np.exp(-d * x) for c, d in zip(self._COEFFS, self._EXPONENTS, strict=False)
                )

                # Energy
                # V(r) = (1/4pi eps0) * Z1*Z2*e^2/r * phi(r/a)
                # In atomic units (Hartree?), or eV?
                # ASE uses eV and Angstroms.
                # Coulomb constant k_e * e^2 = 14.3996 eV * A
                ke_e2 = 14.3996

                v_r = (ke_e2 * z1 * z2 / r) * phi
                energy += v_r

                # Forces
                # F = -dV/dr
                # V = K/r * phi(r/a)
                # dV/dr = -K/r^2 * phi + K/r * (1/a) * dphi/dx
                #       = - (V/r) + (K/r) * (1/a) * dphi_dx

                dv_dr = - (v_r / r) + (ke_e2 * z1 * z2 / r) * (1 / a) * dphi_dx

                # Force on i = -grad_i V = - (dV/dr * dr/dRi)
                # dr/dRi = (Ri - Rj) / r = dist_vec / r
                # F_i = -dv_dr * (dist_vec / r)

                force_on_i = -dv_dr * (dist_vec / r)

                forces[i] += force_on_i
                forces[j] -= force_on_i

        self.results["energy"] = energy
        self.results["forces"] = forces
