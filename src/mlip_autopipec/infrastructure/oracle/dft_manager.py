import concurrent.futures
import copy
import re
from collections.abc import Iterator
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
        self.max_workers = int(self.params.get("max_workers", 1))

        # Security: Validate command
        # Allow alphanumeric, spaces, hyphens, dots, underscores, slashes
        # This is a basic whitelist to prevent shell injection like '; rm -rf /'
        # Note: shlex.split is used by ASE, but we want to be extra safe.
        if self.command and not re.match(r"^[\w\s\-\./]+$", self.command):
            msg = f"Invalid characters in command: {self.command}"
            raise ValueError(msg)

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
                # Log the error here if logging was available in this scope, or rely on caller
                if attempt == max_retries:
                    msg = f"DFT calculation failed after {max_retries} retries."
                    raise RuntimeError(msg) from e
                # Continue to retry

        if result_struct:
            return result_struct

        # Should not reach here
        msg = "DFT Loop Error"
        raise RuntimeError(msg)

    def compute_batch(self, structures: list[Structure]) -> Iterator[Structure]:
        """
        Compute energy and forces for a batch of structures using parallel execution.
        """
        # If max_workers is 1, just use the loop to avoid overhead
        if self.max_workers <= 1:
            for s in structures:
                yield self.compute(s)
            return

        # Using map to preserve order and potentially allow cleaner iteration?
        # No, map blocks until results are ready in order.
        # Use as_completed for streaming, but with bounded submission.

        # For simplicity and to avoid complex bounded queue logic with futures,
        # we can submit in chunks.
        chunk_size = self.max_workers * 2

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            structure_iter = iter(structures)
            while True:
                chunk = []
                try:
                    for _ in range(chunk_size):
                        chunk.append(next(structure_iter))
                except StopIteration:
                    pass

                if not chunk:
                    break

                # Submit chunk
                futures = [executor.submit(self.compute, s) for s in chunk]

                # Yield results as they complete
                for future in concurrent.futures.as_completed(futures):
                    yield future.result()
