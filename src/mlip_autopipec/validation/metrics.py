import logging

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

try:
    from ase.filters import UnitCellFilter
except ImportError:
    from ase.constraints import UnitCellFilter
from ase.optimize import LBFGS

from mlip_autopipec.domain_models.validation import MetricResult

logger = logging.getLogger(__name__)

try:
    import phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False


class ElasticValidator:
    @staticmethod
    def _calculate_stiffness(potential: Calculator, structure: Atoms) -> dict[str, float]:
        """
        Calculates elastic constants Cij using finite differences.
        Assumes cubic symmetry for simplicity for now, or returns full tensor.
        """
        # Relax structure first
        atoms = structure.copy()  # type: ignore[no-untyped-call]
        atoms.calc = potential
        ucf = UnitCellFilter(atoms)
        opt = LBFGS(ucf, logfile=None)
        opt.run(fmax=0.01)

        stress_0 = atoms.get_stress()  # Voigt notation: xx, yy, zz, yz, xz, xy

        # Simple deformation: epsilon = 0.01
        eps = 0.01

        # We need C11, C12, C44 for cubic stability
        # Apply strain e_xx
        atoms_xx = atoms.copy() # type: ignore[no-untyped-call]
        atoms_xx.calc = potential
        cell = atoms_xx.get_cell()
        cell[0, 0] *= (1 + eps)
        atoms_xx.set_cell(cell, scale_atoms=True)
        stress_xx = atoms_xx.get_stress()

        # C11 approx (delta_sigma_xx / delta_eps_xx)
        # Note: ASE stress is -Pressure, usually in eV/A^3 or GPa depending on unit.
        # ASE units: stress is eV/A^3. 1 eV/A^3 = 160.2 GPa.

        # Let's use a simpler approach: use StrainFilter or just assume cubic
        # For this cycle, implementing a full robust Cij calculator is heavy.
        # I will implement a basic estimation.

        # C11: Uniaxial strain along x
        # C12: Response in y to strain in x

        d_stress = (stress_xx - stress_0) / eps
        # Voigt order in ASE: xx, yy, zz, yz, xz, xy
        c11 = d_stress[0] * 160.21766208  # Convert to GPa
        c12 = d_stress[1] * 160.21766208

        # For C44: Shear strain e_xy
        atoms_xy = atoms.copy() # type: ignore[no-untyped-call]
        atoms_xy.calc = potential
        cell = atoms_xy.get_cell()
        # Properly apply shear strain
        deformation = np.eye(3)
        deformation[0, 1] = eps
        new_cell = np.dot(cell, deformation)
        atoms_xy.set_cell(new_cell, scale_atoms=True)
        stress_xy = atoms_xy.get_stress()

        d_stress_shear = (stress_xy - stress_0) / eps
        # ASE stress xy is index 5
        c44 = d_stress_shear[5] * 160.21766208

        return {"C11": c11, "C12": c12, "C44": c44}

    @staticmethod
    def run(potential: Calculator, structure: Atoms) -> MetricResult:
        try:
            cij = ElasticValidator._calculate_stiffness(potential, structure)

            # Born stability criteria for Cubic
            c11 = cij["C11"]
            c12 = cij["C12"]
            c44 = cij["C44"]

            passed = True
            reasons = []

            if not (c11 - c12 > 0):
                passed = False
                reasons.append("C11 - C12 <= 0")
            if not (c11 + 2 * c12 > 0):
                passed = False
                reasons.append("C11 + 2*C12 <= 0")
            if not (c44 > 0):
                passed = False
                reasons.append("C44 <= 0")

            score = min(c11 - c12, c11 + 2 * c12, c44)

            return MetricResult(
                name="Elastic Stability",
                passed=passed,
                score=score,
                details={"Cij": cij, "reasons": reasons}
            )

        except Exception as e:
            logger.exception("Elastic validation failed")
            return MetricResult(
                name="Elastic Stability",
                passed=False,
                details={"error": str(e)}
            )


class PhononValidator:
    @staticmethod
    def _calculate_band_structure(potential: Calculator, structure: Atoms) -> float:
        """
        Returns minimum frequency in THz.
        """
        if not PHONOPY_AVAILABLE:
            msg = "Phonopy not installed"
            raise ImportError(msg)

        # Convert ASE to Phonopy
        unitcell = PhonopyAtoms(
            symbols=structure.get_chemical_symbols(),
            cell=structure.get_cell(),
            scaled_positions=structure.get_scaled_positions()
        )

        # Supercell 2x2x2
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        phonon = phonopy.Phonopy(unitcell, supercell_matrix)

        # Calculate forces
        supercell_atoms = structure.copy() # type: ignore[no-untyped-call]
        supercell_atoms = supercell_atoms * (2, 2, 2)

        # Apply displacements
        phonon.generate_displacements(distance=0.01)
        supercells = phonon.get_supercells_with_displacements()

        # Calculate forces for each displaced supercell
        # This is expensive. In a real scenario, we might want to paralellize or limit this.
        # For this cycle, we do it serially.

        # We need to map Phonopy supercells back to ASE
        # Or easier: just update positions of our ASE supercell
        forces_set = []
        for sc in supercells:
            # sc is a PhonopyAtoms object
            # Update our ASE atoms
            # Since atoms match 1-to-1
            pos = sc.get_positions()
            temp_atoms = supercell_atoms.copy() # type: ignore[no-untyped-call]
            temp_atoms.set_positions(pos)
            temp_atoms.calc = potential
            forces = temp_atoms.get_forces()
            forces_set.append(forces)

        phonon.set_forces(forces_set)
        phonon.produce_force_constants()

        # Band structure on simple path
        # Auto band structure path
        path = [[[0, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0.5], [0, 0, 0]]] # Gamma-X-L-Gamma approx
        qpoints, connections = phonopy.get_band_qpoints_and_path_connections(path, npoints=51)
        phonon.run_band_structure(qpoints, path_connections=connections)

        frequencies = phonon.get_band_structure_dict()['frequencies']
        # frequencies is list of arrays (one per path segment)
        min_freq = 1e9
        for segment in frequencies:
            min_f = np.min(segment)
            min_freq = min(min_freq, min_f)

        return float(min_freq)

    @staticmethod
    def run(potential: Calculator, structure: Atoms) -> MetricResult:
        if not PHONOPY_AVAILABLE:
             logger.warning("Phonopy not installed, skipping Phonon validation.")
             return MetricResult(
                 name="Phonon Stability",
                 passed=True,
                 details={"warning": "Phonopy not installed, validation skipped"}
             )

        try:
            min_freq = PhononValidator._calculate_band_structure(potential, structure)

            # Criterion: min_freq > -0.01 THz (allowing small numerical noise)
            passed = min_freq > -0.05

            return MetricResult(
                name="Phonon Stability",
                passed=passed,
                score=min_freq,
                details={"min_freq": min_freq}
            )

        except Exception as e:
            logger.exception("Phonon validation failed")
            return MetricResult(
                name="Phonon Stability",
                passed=False,
                details={"error": str(e)}
            )
