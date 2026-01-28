import logging
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
except ImportError:
    Phonopy = None
    PhonopyAtoms = None

from mlip_autopipec.config.schemas.validation import PhononConfig
from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult
from mlip_autopipec.validation.utils import load_calculator

logger = logging.getLogger(__name__)


class PhononValidator:
    """
    Validates Phonon spectra/frequencies using Phonopy.
    """

    def __init__(self, config: PhononConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, atoms: Atoms, potential_path: Path) -> ValidationResult:
        logger.info("Starting Phonon Validation...")

        if not self.config.enabled:
            return ValidationResult(module="phonon", passed=True, metrics=[], error=None)

        if Phonopy is None:
             return ValidationResult(
                module="phonon",
                passed=False,
                error="Phonopy not installed"
            )

        try:
            # Load calculator
            calc = load_calculator(potential_path)

            # Convert ASE to PhonopyAtoms
            unitcell = PhonopyAtoms(
                symbols=atoms.get_chemical_symbols(),
                scaled_positions=atoms.get_scaled_positions(),
                cell=atoms.get_cell(),
            )

            # Initialize Phonopy
            phonon = Phonopy(
                unitcell,
                supercell_matrix=self.config.supercell_matrix,
                symprec=self.config.symprec
            )

            # Generate displacements
            phonon.generate_displacements(distance=self.config.displacement)
            supercells = phonon.supercells_with_displacements

            if supercells is None:
                msg = "Failed to generate supercells"
                raise RuntimeError(msg)

            # Calculate forces
            sets_of_forces = []
            for sc in supercells:
                # Convert back to ASE for calculation
                # PhonopyAtoms attributes: symbols, scaled_positions, cell
                sc_ase = Atoms(
                    symbols=sc.symbols,
                    scaled_positions=sc.scaled_positions,
                    cell=sc.cell,
                    pbc=True
                )
                sc_ase.calc = calc
                forces = sc_ase.get_forces()
                sets_of_forces.append(forces)

            # Produce force constants
            phonon.forces = sets_of_forces
            phonon.produce_force_constants()

            # Check for imaginary frequencies on a mesh
            # Using a dense mesh to sample BZ
            mesh = [20, 20, 20]
            phonon.run_mesh(mesh, with_eigenvectors=False, is_gamma_center=True)
            mesh_dict = phonon.get_mesh_dict()
            frequencies = mesh_dict["frequencies"]

            # Flatten frequencies
            freqs = frequencies.flatten()

            # Filter out acoustic modes at Gamma (usually 3 smallest absolute values close to 0)
            # Actually, just checking min freq is usually enough.
            # Imaginary frequencies are returned as negative real numbers in older phonopy,
            # or complex numbers?
            # Standard Phonopy returns real numbers where negative value means imaginary freq squared is negative.
            # i.e. omega^2 < 0 => omega is imaginary. Phonopy returns sign(omega^2) * sqrt(abs(omega^2)).
            # So negative frequency means instability.

            # We ignore very small negative numbers due to numerical noise
            min_freq = np.min(freqs)

            # Threshold for instability: slightly below zero
            threshold = -0.05 # THz (or whatever unit, usually THz)

            passed = min_freq > threshold

            details: dict[str, Any] = {
                "min_frequency": float(min_freq),
                "threshold": threshold,
                "n_imaginary": int(np.sum(freqs < threshold))
            }

            if not passed:
                details["status"] = "Imaginary frequencies detected (Unstable)"
            else:
                details["status"] = "Stable"

            metric = ValidationMetric(
                name="min_frequency",
                value=float(min_freq),
                unit="THz",
                passed=passed,
                details=details
            )

            # Save band structure plot if failed? Or always?
            # Creating plot requires more logic, maybe leave for report generator using data?
            # Or save yaml/hdf5
            phonon.save(self.work_dir / "phonon.yaml")

            return ValidationResult(
                module="phonon",
                passed=passed,
                metrics=[metric],
                error=None
            )

        except Exception as e:
            logger.exception("Phonon validation failed")
            return ValidationResult(
                module="phonon",
                passed=False,
                error=str(e)
            )
