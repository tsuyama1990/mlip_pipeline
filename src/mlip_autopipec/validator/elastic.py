import logging
from typing import NamedTuple

import numpy as np
from ase import Atoms

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.dynamics.calculators import MLIPCalculatorFactory

logger = logging.getLogger(__name__)

class ElasticResults(NamedTuple):
    """
    Elastic constants results (in GPa).

    Attributes:
        C11: Elastic constant C11 (GPa)
        C12: Elastic constant C12 (GPa)
        C44: Elastic constant C44 (GPa)
        B: Bulk modulus (GPa)
        G: Shear modulus (GPa)
    """
    C11: float
    C12: float
    C44: float
    B: float
    G: float

class ElasticAnalyzer:
    """Calculates elastic constants for cubic crystals."""

    def __init__(self, delta: float = 0.01) -> None:
        """
        Initialize ElasticAnalyzer.

        Args:
            delta: Strain magnitude to apply.
        """
        self.delta = delta
        self.conversion_factor = 160.21766208  # eV/A^3 to GPa

    def _apply_strain_and_get_stress(self, atoms: Atoms, deformation: np.ndarray) -> np.ndarray:
        """Apply deformation and return stress tensor."""
        new_atoms = atoms.copy() # type: ignore[no-untyped-call]
        new_atoms.calc = atoms.calc

        initial_cell = atoms.get_cell() # type: ignore[no-untyped-call]
        # Matrix multiplication: new_cell = initial_cell @ deformation.T
        new_cell = np.dot(initial_cell, deformation.T)
        new_atoms.set_cell(new_cell, scale_atoms=True)
        # ASE get_stress returns Any or ndarray
        return new_atoms.get_stress(voigt=False) # type: ignore[no-any-return]

    def _calculate_c11_c12(self, atoms: Atoms) -> tuple[float, float]:
        """Calculate C11 and C12 using tensile strain."""
        delta = self.delta

        # Strain positive delta in xx
        def_pos = np.eye(3)
        def_pos[0,0] += delta
        stress_pos = self._apply_strain_and_get_stress(atoms, def_pos)

        # Strain negative delta in xx
        def_neg = np.eye(3)
        def_neg[0,0] -= delta
        stress_neg = self._apply_strain_and_get_stress(atoms, def_neg)

        # Finite difference
        # We assume ASE returns -stress (virial). Negate it.
        ds_de = -(stress_pos - stress_neg) / (2 * delta)

        c11 = ds_de[0, 0]
        c12 = ds_de[1, 1]
        return float(c11), float(c12)

    def _calculate_c44(self, atoms: Atoms) -> float:
        """Calculate C44 using shear strain."""
        delta = self.delta

        # Strain positive delta/2 (shear gamma = delta) in yz
        eps_shear = np.zeros((3,3))
        eps_shear[1,2] = delta/2
        eps_shear[2,1] = delta/2
        def_pos = np.eye(3) + eps_shear
        stress_pos = self._apply_strain_and_get_stress(atoms, def_pos)

        # Strain negative delta/2 (shear gamma = -delta) in yz
        eps_shear_neg = np.zeros((3,3))
        eps_shear_neg[1,2] = -delta/2
        eps_shear_neg[2,1] = -delta/2
        def_neg = np.eye(3) + eps_shear_neg
        stress_neg = self._apply_strain_and_get_stress(atoms, def_neg)

        ds_shear = -(stress_pos - stress_neg)
        # d_sigma_yz / d_gamma_yz
        c44 = ds_shear[1, 2] / (2 * delta)
        return float(c44)

    def calculate_elastic_constants(self, structure: Atoms, potential: Potential) -> ElasticResults:
        """
        Calculate elastic constants C11, C12, C44 for a cubic structure.

        Args:
            structure: ASE Atoms object (cubic).
            potential: Potential to use.

        Returns:
            ElasticResults object.
        """
        atoms = structure.copy() # type: ignore[no-untyped-call]
        # Instantiate factory and create calculator
        factory = MLIPCalculatorFactory()
        # But here 'potential' is domain_models.potential.Potential
        # Let's check Potential model.
        if potential.path is None:
            msg = "Potential path is not set."
            raise ValueError(msg)

        calc = factory.create(potential.path)
        atoms.calc = calc

        c11, c12 = self._calculate_c11_c12(atoms)
        c44 = self._calculate_c44(atoms)

        # Convert to GPa
        c11_gpa = c11 * self.conversion_factor
        c12_gpa = c12 * self.conversion_factor
        c44_gpa = c44 * self.conversion_factor

        b_gpa = (c11_gpa + 2 * c12_gpa) / 3
        # Voigt average shear modulus for cubic
        g_v = (c11_gpa - c12_gpa + 3 * c44_gpa) / 5

        # Reuss average
        denom = (4 / (c11_gpa - c12_gpa) + 3 / c44_gpa)
        if abs(denom) > 1e-9 and abs(c44_gpa) > 1e-9 and abs(c11_gpa - c12_gpa) > 1e-9:
            g_r = 5 / denom
        else:
            g_r = 0.0 # Handle instability

        g_h = (g_v + g_r) / 2

        return ElasticResults(
            C11=c11_gpa,
            C12=c12_gpa,
            C44=c44_gpa,
            B=b_gpa,
            G=g_h
        )
