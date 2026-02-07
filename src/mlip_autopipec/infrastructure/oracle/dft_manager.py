import copy
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import CalculationFailed
from ase.calculators.espresso import Espresso, EspressoProfile

from mlip_autopipec.domain_models import Structure
from mlip_autopipec.interfaces import BaseOracle
from mlip_autopipec.utils.physics import kspacing_to_grid


class DFTManager(BaseOracle):
    """
    Oracle implementation using ASE and Quantum Espresso.
    """
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        # Parse params
        self.command = self.params.get("command")
        self.pseudo_dir = self.params.get("pseudo_dir")
        self.pseudopotentials = self.params.get("pseudopotentials")
        self.kspacing = float(self.params.get("kspacing", 0.04))
        self.smearing_width = float(self.params.get("smearing_width", 0.02))

    def compute(self, structure: Structure) -> Structure:
        """
        Compute energy and forces for the given structure.
        """
        # Convert to ASE Atoms
        atoms = Atoms(
            symbols=structure.species,
            positions=structure.positions,
            cell=structure.cell,
            pbc=True
        )

        # Calculate k-points
        kpoints = kspacing_to_grid(structure.cell, self.kspacing)

        # Parse command for EspressoProfile
        command_str = self.command or "pw.x"

        pseudo_dir_str = str(self.pseudo_dir) if self.pseudo_dir else "."

        # EspressoProfile requires command string and pseudo_dir
        # ase untyped
        profile = EspressoProfile(command=command_str, pseudo_dir=pseudo_dir_str) # type: ignore[no-untyped-call]

        # Base parameters
        base_input_data: dict[str, Any] = {
            "control": {
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "pseudo_dir": pseudo_dir_str,
                "tprnfor": True,  # Forces
                "tstress": True,  # Stress
            },
            "system": {
                "ecutwfc": 40.0,
                "occupations": "smearing",
                "smearing": "gaussian",
                "degauss": self.smearing_width,
            },
            "electrons": {
                "mixing_beta": 0.7,
                "conv_thr": 1.0e-6,
            }
        }

        # Self-healing loop
        max_retries = 2

        result_struct = None

        for attempt in range(max_retries + 1):
            try:
                # Deep copy to ensure we don't mutate base_input_data across attempts
                # and to isolate attempts for testing/references.
                current_input_data = copy.deepcopy(base_input_data)

                # Adjust mixing_beta if retrying
                if attempt > 0:
                    # Fix indexed assignment error by ensuring type safety
                    # current_input_data["electrons"] is Any (dict), so safe.
                    current_input_data["electrons"]["mixing_beta"] = 0.35  # Reduce by 50%

                calc = Espresso( # type: ignore[no-untyped-call]
                    profile=profile,
                    pseudopotentials=self.pseudopotentials,
                    kpts=kpoints,
                    input_data=current_input_data,
                )
                atoms.calc = calc

                energy = atoms.get_potential_energy() # type: ignore[no-untyped-call]
                forces = atoms.get_forces() # type: ignore[no-untyped-call]
                stress_voigt = atoms.get_stress() # type: ignore[no-untyped-call]

                # Convert Voigt (6,) to tensor (3,3)
                if stress_voigt.shape == (6,):
                    stress = np.array([
                        [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
                        [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
                        [stress_voigt[4], stress_voigt[3], stress_voigt[2]]
                    ])
                else:
                    stress = stress_voigt

                # Success
                result_struct = structure.model_copy(deep=True)
                result_struct.energy = float(energy)
                result_struct.forces = forces
                result_struct.stress = stress
                break # Exit loop on success

            except CalculationFailed as e:
                if attempt == max_retries:
                    msg = f"DFT calculation failed after {max_retries} retries."
                    raise RuntimeError(msg) from e
                # Continue to retry

        if result_struct:
            return result_struct

        # Should not reach here
        msg = "DFT Loop Error"
        raise RuntimeError(msg)
