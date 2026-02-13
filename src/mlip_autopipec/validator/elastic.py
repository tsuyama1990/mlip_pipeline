import logging
import numpy as np
from ase.optimize import LBFGS
from typing import NamedTuple, Any

# ASE 3.24+ moves CellFilter to ase.filters
try:
    from ase.filters import ExpCellFilter, UnitCellFilter
except ImportError:
    # Older ASE or failure
    # Try importing individually to identify issue
    try:
        from ase.filters import ExpCellFilter
        from ase.filters import UnitCellFilter
    except ImportError:
        # Fallback to constraints
        from ase.constraints import ExpCellFilter, UnitCellFilter # type: ignore

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.dynamics.calculators import MLIPCalculatorFactory

logger = logging.getLogger(__name__)

class ElasticResults(NamedTuple):
    C11: float
    C12: float
    C44: float
    bulk_modulus: float
    shear_modulus: float

class ElasticAnalyzer:
    """Analyzer for Elastic Constants."""

    def __init__(self, strain_magnitude: float = 0.01) -> None:
        self.strain_magnitude = strain_magnitude
        self.calculator_factory = MLIPCalculatorFactory()

    def analyze(self, potential: Potential, structure: Structure) -> dict[str, float]:
        """
        Calculates elastic constants.

        Args:
            potential: The potential to use.
            structure: The structure to analyze.

        Returns:
            Dictionary with elastic constants and moduli (GPa).
        """
        atoms = structure.to_ase().copy() # type: ignore[no-untyped-call]

        # Setup calculator
        calc = self.calculator_factory.create(potential.path)
        atoms.calc = calc

        # 1. Relax structure first
        try:
            # Prefer UnitCellFilter
            try:
                ucf = UnitCellFilter(atoms) # type: ignore[no-untyped-call]
            except NameError:
                ucf = ExpCellFilter(atoms)

            opt = LBFGS(ucf, logfile=None) # type: ignore[arg-type]
            opt.run(fmax=0.01) # type: ignore[no-untyped-call]
        except Exception:
            logger.warning("Relaxation failed. Proceeding with unrelaxed structure.", exc_info=True)

        # 2. Apply strains and compute stress
        delta = self.strain_magnitude

        # Conversion: eV/A^3 to GPa
        EV_A3_TO_GPA = 160.21766208

        # Helper to get stress
        def get_stress(strained_atoms: Any) -> Any:
            # Returns stress in Voigt notation: [sxx, syy, szz, syz, sxz, sxy]
            # Units are eV per Angstrom cubed
            return strained_atoms.get_stress(voigt=True)

        # Baseline
        base_stress = get_stress(atoms)
        cell = atoms.get_cell()

        # 1. C11 and C12
        # Apply tensile strain in x: e_xx = delta
        atoms_c11 = atoms.copy()

        # strain tensor: [[1+delta, 0, 0], [0, 1, 0], [0, 0, 1]]
        distortion_c11 = np.eye(3)
        distortion_c11[0, 0] += delta

        # Apply deformation to cell vectors
        # If cell vectors are rows: v' = v @ F
        new_cell_c11 = np.dot(cell, distortion_c11)

        atoms_c11.set_cell(new_cell_c11, scale_atoms=True)
        atoms_c11.calc = self.calculator_factory.create(potential.path)
        stress_c11 = get_stress(atoms_c11)

        c11 = (stress_c11[0] - base_stress[0]) / delta
        c12 = (stress_c11[1] - base_stress[1]) / delta

        # 2. C44
        # Apply shear strain in yz: gamma_yz = delta
        atoms_c44 = atoms.copy()

        epsilon = np.zeros((3, 3))
        epsilon[1, 2] = delta / 2
        epsilon[2, 1] = delta / 2

        distortion_c44 = np.eye(3) + epsilon
        # Apply deformation to cell vectors
        new_cell_c44 = np.dot(cell, distortion_c44)

        atoms_c44.set_cell(new_cell_c44, scale_atoms=True)
        atoms_c44.calc = self.calculator_factory.create(potential.path)
        stress_c44 = get_stress(atoms_c44)

        # sigma_yz = C44 * gamma_yz = C44 * delta
        c44 = (stress_c44[3] - base_stress[3]) / delta # index 3 is yz

        # Convert to GPa
        c11_gpa = c11 * EV_A3_TO_GPA
        c12_gpa = c12 * EV_A3_TO_GPA
        c44_gpa = c44 * EV_A3_TO_GPA

        bulk_modulus = (c11_gpa + 2 * c12_gpa) / 3
        shear_modulus = c44_gpa

        return {
            "C11": c11_gpa,
            "C12": c12_gpa,
            "C44": c44_gpa,
            "bulk_modulus": bulk_modulus,
            "shear_modulus": shear_modulus
        }
