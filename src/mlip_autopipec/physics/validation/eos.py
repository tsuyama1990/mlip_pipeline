from pathlib import Path

import numpy as np
from ase.eos import EquationOfState

from mlip_autopipec.domain_models.config import ValidationConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult
from .utils import get_lammps_calculator


class EOSValidator:
    def __init__(self, config: ValidationConfig, potential_path: Path):
        self.config = config
        self.potential_path = potential_path

    def validate(self, structure: Structure) -> ValidationResult:
        atoms = structure.to_ase()
        # Assume single element for now or get from structure
        elements = sorted(list(set(atoms.get_chemical_symbols())))
        calc = get_lammps_calculator(self.potential_path, elements)

        # Generate volumes
        cell = atoms.get_cell()
        volumes = []
        energies = []

        # +/- range (5 points)
        # We start from 1.0.
        # range is e.g. 0.2 -> 0.8 to 1.2
        factors = np.linspace(
            1.0 - self.config.eos_vol_range, 1.0 + self.config.eos_vol_range, 5
        )

        for x in factors:
            atoms_strain = atoms.copy()  # type: ignore[no-untyped-call]
            atoms_strain.calc = calc
            # scale_atoms=True scales positions
            atoms_strain.set_cell(
                cell * (x ** (1 / 3)), scale_atoms=True
            )  # x is volume scale, so linear scale is x^(1/3)

            # Recalculate cell to be sure? set_cell updates it.

            try:
                e = atoms_strain.get_potential_energy()
                v = atoms_strain.get_volume()
            except Exception:
                # If calculation fails (e.g. lost atoms, lammps error), skip or fail
                # For robustness, we fail
                return ValidationResult(
                    potential_id=str(self.potential_path),
                    metrics=[],
                    plots={},
                    overall_status="FAIL",
                )

            energies.append(e)
            volumes.append(v)

        # Fit EOS
        try:
            eos = EquationOfState(volumes, energies)
            e0, B, Bprime, v0 = eos.fit()  # type: ignore[no-untyped-call]

            # B is in eV/A^3. Convert to GPa?
            # 1 eV/A^3 = 160.21766208 GPa
            B_GPa = B * 160.21766208

            passed = bool(B > 0)

            metric = ValidationMetric(
                name="Bulk Modulus", value=float(B_GPa), passed=passed
            )

            return ValidationResult(
                potential_id=str(self.potential_path),
                metrics=[metric],
                plots={},
                overall_status="PASS" if passed else "FAIL",
            )

        except Exception:
            return ValidationResult(
                potential_id=str(self.potential_path),
                metrics=[],
                plots={},
                overall_status="FAIL",
            )
