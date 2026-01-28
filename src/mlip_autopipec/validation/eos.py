import numpy as np
from ase import Atoms
from ase.eos import EquationOfState
from ase.units import GPa

from mlip_autopipec.config.schemas.validation import EOSConfig
from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult


class EOSValidator:
    """
    Validates Equation of State (EOS) and calculates Bulk Modulus.
    """

    def __init__(self, config: EOSConfig) -> None:
        self.config = config

    def validate(self, atoms: Atoms) -> ValidationResult:
        """
        Fit Birch-Murnaghan EOS and check Bulk Modulus.

        Args:
            atoms: ASE Atoms object with calculator.

        Returns:
            ValidationResult with Bulk Modulus.
        """
        if not isinstance(atoms, Atoms):
            return ValidationResult(
                module="eos", passed=False, error=f"Expected ase.Atoms object, got {type(atoms)}"
            )

        if atoms.calc is None:
            return ValidationResult(
                module="eos", passed=False, error="Atoms object has no calculator attached."
            )

        try:
            volumes = []
            energies = []

            # Generate volume scaling factors
            # strain_max is max volumetric strain e.g. 0.1 means +/- 10% volume
            scale_factors = np.linspace(
                1.0 - self.config.strain_max, 1.0 + self.config.strain_max, self.config.num_points
            )

            for s in scale_factors:
                temp = atoms.copy()
                temp.calc = atoms.calc

                # Scale volume by s -> linear scale by s^(1/3)
                linear_scale = s ** (1 / 3)
                temp.set_cell(atoms.get_cell() * linear_scale, scale_atoms=True)

                energies.append(temp.get_potential_energy())
                volumes.append(temp.get_volume())

            # Fit EOS
            eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
            v0_fit, e0_fit, B_fit = eos.fit()

            # Convert B to GPa (B_fit is in eV/A^3)
            # ase.units.GPa is actually 1/160.2... ?
            # No, ase.units.GPa is the value of 1 GPa in atomic units (eV, Ang).
            # 1 GPa = 0.00624 eV/A^3.
            # So B (in eV/A^3) / GPa -> B (in GPa).
            B_GPa = B_fit / GPa

            # Stability check: Bulk modulus must be positive
            passed = B_GPa > 0.0

            metrics = [
                ValidationMetric(name="bulk_modulus", value=B_GPa, unit="GPa", passed=passed),
                ValidationMetric(name="equilibrium_volume", value=v0_fit, unit="A^3", passed=True),
                ValidationMetric(name="min_energy", value=e0_fit, unit="eV", passed=True),
            ]

            return ValidationResult(module="eos", passed=passed, metrics=metrics)

        except Exception as e:
            return ValidationResult(
                module="eos", passed=False, error=f"EOS calculation failed: {e!s}"
            )
