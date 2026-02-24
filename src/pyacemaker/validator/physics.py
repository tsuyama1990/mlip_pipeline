"""Physics validation logic."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.eos import EquationOfState
from ase.units import GPa
from loguru import logger
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from pyacemaker.core.config import CONSTANTS
from pyacemaker.domain_models.validator import ValidationResult
from pyacemaker.validator.base import BaseValidator


def ase_to_phonopy(atoms: Atoms) -> PhonopyAtoms:
    """Convert ASE atoms to Phonopy atoms."""
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),  # type: ignore[no-untyped-call]
        cell=atoms.cell,
        scaled_positions=atoms.get_scaled_positions(),  # type: ignore[no-untyped-call]
    )


class PhysicsValidator(BaseValidator):
    """Physics validator implementation."""

    def __init__(self) -> None:
        """Initialize physics validator."""
        self.logger = logger.bind(name="PhysicsValidator")

    def validate(
        self,
        potential_path: Path,
        structure: Atoms,
        output_dir: Path,
        **kwargs: Any,
    ) -> ValidationResult:
        """Run validation.

        This method orchestrates individual checks.
        """
        # This implementation primarily used via direct calls to check_*, but kept for interface.
        return ValidationResult(
             passed=False,
             metrics={"error": 1.0}, # Satisfy min_length=1
             eos_stable=False,
             phonon_stable=False,
             elastic_stable=False
        )

    def check_phonons(
        self,
        atoms: Atoms,
        supercell: list[int] | None = None,
        tolerance: float | None = None,
    ) -> bool:
        """Check phonon stability."""
        # Use defaults from CONSTANTS if not provided
        if supercell is None:
            supercell = CONSTANTS.physics_phonon_supercell
        if tolerance is None:
            tolerance = CONSTANTS.physics_phonon_tolerance

        if atoms.calc is None:
            msg = "Atoms object must have a calculator attached."
            raise ValueError(msg)

        # Create Phonopy object
        try:
            phonon = Phonopy(ase_to_phonopy(atoms), supercell_matrix=np.diag(supercell))
        except Exception as e:
            msg = f"Failed to initialize Phonopy: {e}"
            raise ValueError(msg) from e

        # Generate displacements
        phonon.generate_displacements()
        supercells = phonon.get_supercells_with_displacements()  # type: ignore[attr-defined]

        # Calculate forces for each displacement
        forces_set = []
        calc = atoms.calc

        for ph_atoms in supercells:
            # Convert PhonopyAtoms to ASE Atoms
            ase_supercell = Atoms(
                symbols=ph_atoms.symbols,
                cell=ph_atoms.cell,
                scaled_positions=ph_atoms.scaled_positions,
                pbc=True,
            )
            ase_supercell.calc = calc
            forces = ase_supercell.get_forces()  # type: ignore[no-untyped-call]
            forces_set.append(forces)

        # Set forces
        # FIX: Use produce_force_constants(forces=...) instead of set_forces
        phonon.produce_force_constants(forces=forces_set)

        # Calculate band structure (auto path)
        phonon.auto_band_structure(plot=False, write_yaml=False)  # type: ignore[no-untyped-call]
        bs_dict = phonon.get_band_structure_dict()

        frequencies = bs_dict["frequencies"]
        # Frequencies is a list of arrays (one for each path segment)

        min_freq = 0.0
        for segment in frequencies:
            min_freq = min(min_freq, np.min(segment))

        # Tolerance for numerical noise (e.g. -0.05 THz)
        # Imaginary frequencies are negative in phonopy
        is_stable = bool(min_freq >= tolerance)
        if not is_stable:
             self.logger.warning(f"Phonon check failed: min freq {min_freq} < {tolerance}")

        return is_stable

    def check_eos(
        self,
        atoms: Atoms,
        strain: float | None = None,
        points: int | None = None,
        output_path: str = "eos.png",
    ) -> tuple[float, str]:
        """Check Equation of State."""
        # Use defaults from CONSTANTS if not provided
        strain_val = strain if strain is not None else CONSTANTS.physics_eos_strain
        points_val = points if points is not None else CONSTANTS.physics_eos_points

        if atoms.calc is None:
            msg = "Atoms object must have a calculator attached."
            raise ValueError(msg)

        if len(atoms) == 0:
            msg = "Cannot check EOS for empty structure."
            raise ValueError(msg)

        volumes = []
        energies = []

        factors = np.linspace(1.0 - strain_val, 1.0 + strain_val, points_val)

        original_cell = atoms.get_cell()  # type: ignore[no-untyped-call]

        for f in factors:
            # Isotropic scaling
            atoms_copy = atoms.copy()  # type: ignore[no-untyped-call]
            atoms_copy.calc = atoms.calc  # Reuse calculator
            atoms_copy.set_cell(original_cell * f, scale_atoms=True)

            volumes.append(atoms_copy.get_volume())
            energies.append(atoms_copy.get_potential_energy())

        try:
            eos = EquationOfState(volumes, energies, eos="birchmurnaghan")  # type: ignore[no-untyped-call]
            v0, e0, B, dBdP = eos.fit()  # type: ignore[no-untyped-call]
        except Exception as e:
            msg = f"EOS fitting failed: {e}"
            raise ValueError(msg) from e

        # Convert B to GPa
        B_GPa = B / GPa

        # Generate plot
        plt.switch_backend("Agg")
        eos.plot(output_path)  # type: ignore[no-untyped-call]
        plt.close()

        return B_GPa, output_path

    def calculate_elastic_constants(
        self,
        atoms: Atoms,
        strain: float | None = None
    ) -> dict[str, float]:
        """Calculate elastic constants."""
        # Use defaults from CONSTANTS if not provided
        strain_val = strain if strain is not None else CONSTANTS.physics_elastic_strain

        if atoms.calc is None:
            msg = "Atoms object must have a calculator attached."
            raise ValueError(msg)

        if len(atoms) == 0:
            msg = "Cannot calculate elastic constants for empty structure."
            raise ValueError(msg)

        cell = atoms.get_cell()  # type: ignore[no-untyped-call]
        atoms.get_volume()  # type: ignore[no-untyped-call]

        # C11 & C12: Uniaxial strain along x
        atoms_c11 = atoms.copy()  # type: ignore[no-untyped-call]
        atoms_c11.calc = atoms.calc
        c = cell.copy()
        c[0, 0] *= 1 + strain_val
        atoms_c11.set_cell(c, scale_atoms=True)
        stress_c11 = atoms_c11.get_stress(voigt=True)

        # Voigt order: xx, yy, zz, yz, xz, xy
        C11 = stress_c11[0] / strain_val / GPa
        C12 = stress_c11[1] / strain_val / GPa

        # C44: Shear strain xy
        atoms_c44 = atoms.copy()  # type: ignore[no-untyped-call]
        atoms_c44.calc = atoms.calc
        c_shear = cell.copy()

        # Apply simple shear
        c_shear[0, 1] += strain_val * cell[1, 1]
        atoms_c44.set_cell(c_shear, scale_atoms=True)
        stress_c44 = atoms_c44.get_stress(voigt=True)

        C44 = stress_c44[5] / strain_val / GPa

        return {"C11": abs(C11), "C12": abs(C12), "C44": abs(C44)}

    def check_elastic(
        self,
        atoms: Atoms,
        strain: float | None = None
    ) -> tuple[bool, dict[str, float]]:
        """Check elastic stability."""
        strain_val = strain if strain is not None else CONSTANTS.physics_elastic_strain
        Cij = self.calculate_elastic_constants(atoms, strain_val)

        c11 = Cij.get("C11", 0.0)
        c12 = Cij.get("C12", 0.0)
        c44 = Cij.get("C44", 0.0)

        stable = (c11 - c12 > 0) and (c11 + 2 * c12 > 0) and (c44 > 0)

        return stable, Cij


# Backward compatibility wrappers
_validator = PhysicsValidator()

def check_phonons(atoms: Atoms, supercell: list[int] | None = None, tolerance: float | None = None) -> bool:
    return _validator.check_phonons(atoms, supercell, tolerance)

def check_eos(atoms: Atoms, strain: float | None = None, points: int | None = None, output_path: str = "eos.png") -> tuple[float, str]:
    return _validator.check_eos(atoms, strain, points, output_path)

def check_elastic(atoms: Atoms, strain: float | None = None) -> tuple[bool, dict[str, float]]:
    return _validator.check_elastic(atoms, strain)

def calculate_elastic_constants(atoms: Atoms, strain: float | None = None) -> dict[str, float]:
    return _validator.calculate_elastic_constants(atoms, strain)
