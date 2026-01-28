import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from mlip_autopipec.config.schemas.validation import PhononConfig
from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult


class PhononValidator:
    """
    Validates phonon stability using Phonopy.
    Checks for imaginary frequencies in the phonon dispersion.
    """

    def __init__(self, config: PhononConfig) -> None:
        self.config = config

    def validate(self, atoms: Atoms) -> ValidationResult:
        """
        Run phonon validation.

        Args:
            atoms: ASE Atoms object with a calculator attached.

        Returns:
            ValidationResult containing stability metrics.
        """
        if not isinstance(atoms, Atoms):
            return ValidationResult(
                module="phonon", passed=False, error=f"Expected ase.Atoms object, got {type(atoms)}"
            )

        if atoms.calc is None:
            return ValidationResult(
                module="phonon", passed=False, error="Atoms object has no calculator attached."
            )

        try:
            # 1. Convert ASE to PhonopyAtoms
            unitcell = PhonopyAtoms(
                symbols=atoms.get_chemical_symbols(),
                cell=atoms.get_cell(),
                scaled_positions=atoms.get_scaled_positions(),
            )

            # 2. Initialize Phonopy
            phonon = Phonopy(
                unitcell,
                supercell_matrix=self.config.supercell_matrix,
                factor=1.0,  # Default VASP-like factor
                symprec=self.config.symprec,
            )

            # 3. Generate displacements
            phonon.generate_displacements(distance=self.config.displacement)
            supercells = phonon.supercells_with_displacements

            # 4. Calculate forces
            forces_set = []
            for sc in supercells:
                # Convert back to ASE
                sc_ase = Atoms(
                    symbols=sc.symbols, scaled_positions=sc.scaled_positions, cell=sc.cell, pbc=True
                )

                # Attach calculator (Reuse the one from input atoms)
                # Note: This assumes the calculator is compatible with the supercell (e.g. MLIP)
                # and doesn't rely on fixed system size initialization.
                sc_ase.calc = atoms.calc

                forces = sc_ase.get_forces()
                forces_set.append(forces)

            # 5. Set forces and produce force constants
            phonon.set_forces(forces_set)
            phonon.produce_force_constants()

            # 6. Check stability using Mesh (faster and covers BZ)
            # Use a dense mesh for checking
            mesh_density = [20, 20, 20]
            phonon.run_mesh(mesh=mesh_density)
            mesh_dict = phonon.get_mesh_dict()
            frequencies = mesh_dict["frequencies"]  # Shape: (num_qpoints, num_bands)

            # 7. Analyze results
            min_freq = float(np.min(frequencies))

            # Tolerance for translational invariance breaking (gamma point acoustic modes)
            # Usually small negative values are allowed.
            TOLERANCE = -0.1  # THz

            passed = min_freq > TOLERANCE

            metrics = [
                ValidationMetric(
                    name="min_frequency",
                    value=min_freq,
                    unit="THz",
                    passed=passed,
                    details={"tolerance": TOLERANCE},
                ),
                ValidationMetric(name="is_stable", value=passed, passed=passed),
            ]

            return ValidationResult(module="phonon", passed=passed, metrics=metrics)

        except Exception as e:
            return ValidationResult(
                module="phonon", passed=False, error=f"Phonon calculation failed: {e!s}"
            )
