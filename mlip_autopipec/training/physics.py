import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class ZBLCalculator(Calculator):
    """
    Calculates ZBL (Ziegler-Biersack-Littmark) baseline energy and forces.

    This acts as a screened Coulomb potential for short-range repulsion.
    V(r) = (Z1*Z2*e^2/r) * phi(r/a)

    We use a simplified implementation or wrap an existing one if available.
    For this cycle, we implement the analytical form.
    """

    implemented_properties = ["energy", "forces"]

    # Universal screening function coefficients (phi)
    _COEFFS = [0.1818, 0.5099, 0.2802, 0.02817]
    _EXPONENTS = [3.2, 0.9423, 0.4029, 0.2016]

    def calculate(self, atoms: Atoms, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        energy = 0.0
        forces = np.zeros((len(atoms), 3))

        # Simple N^2 loop for pairwise interactions
        # Note: In production, this should be optimized (neighbor list)
        # But for ZBL, cutoff is usually short, or we rely on the fact that
        # it decays very fast.

        positions = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()

        # Constants
        # e^2 in eV*Angstrom = 14.3996
        KE2 = 14.3996

        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                dist_vec = positions[i] - positions[j]
                r = np.linalg.norm(dist_vec)

                if r < 1e-4: # Avoid singularity
                    continue

                z1 = numbers[i]
                z2 = numbers[j]

                # Screening length a
                # a = 0.8854 * a0 / (Z1^0.23 + Z2^0.23)
                # a0 (Bohr radius) = 0.529 Angstroms
                a = (0.8854 * 0.529) / (z1**0.23 + z2**0.23)

                x = r / a
                phi = sum(c * np.exp(-d * x) for c, d in zip(self._COEFFS, self._EXPONENTS))
                dphi_dx = sum(-c * d * np.exp(-d * x) for c, d in zip(self._COEFFS, self._EXPONENTS))

                # V = (Z1 Z2 e^2 / r) * phi
                coulomb = (z1 * z2 * KE2) / r
                v_pair = coulomb * phi

                energy += v_pair

                # Forces
                # F = -dV/dr
                # V = K/r * phi(r/a)
                # dV/dr = -K/r^2 * phi + K/r * (1/a) * dphi/dx

                dv_dr = -v_pair / r + (coulomb / a) * dphi_dx

                f_vec = (dv_dr * dist_vec) / r # Force on i due to j (repulsive -> positive dv_dr means attraction? Wait.)
                # Repulsive potential: V decreases as r increases -> dV/dr < 0.
                # Here v_pair is positive.
                # Term 1: -v_pair/r is negative.
                # Term 2: dphi_dx is negative. So term 2 is negative.
                # So dv_dr is negative.
                # Force on i = -grad_i V = - (dV/dr * dr/dRi) = - dV/dr * (Ri - Rj)/r
                # F_i = -dv_dr * (Ri - Rj) / r

                # Let's check sign.
                # Repulsion: Force on i should be along (Ri - Rj).
                # If dv_dr is negative, -dv_dr is positive. So F_i is along (Ri - Rj). Correct.

                force_on_i = -dv_dr * (dist_vec / r)

                forces[i] += force_on_i
                forces[j] -= force_on_i

        self.results["energy"] = energy
        self.results["forces"] = forces
