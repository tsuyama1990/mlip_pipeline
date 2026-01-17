from typing import Any

import numpy as np
from ase import Atoms

from mlip_autopipec.dft.constants import (
    DEFAULT_KPOINT_DENSITY,
    MAGNETIC_ELEMENTS,
    SSSP_EFFICIENCY_1_1,
)


class InputGenerator:
    """
    Generates Quantum Espresso input files (pw.in) based on physical rules.
    """

    @staticmethod
    def create_input_string(atoms: Atoms, params: dict[str, Any] | None = None) -> str:
        """
        Generates the content of a pw.in file.

        Args:
            atoms: The ASE Atoms object.
            params: Optional overrides for parameters.
        """
        if params is None:
            params = {}

        # 1. Determine K-points based on density
        k_points = InputGenerator._calculate_kpoints(
            atoms, params.get("k_density", DEFAULT_KPOINT_DENSITY)
        )

        # 2. Determine Magnetism
        magnetism_settings = InputGenerator._determine_magnetism(atoms)

        # 3. Determine Pseudopotentials
        pseudopotentials = InputGenerator._get_pseudopotentials(atoms)

        # 4. Construct the input string (Manual construction or via ASE's write_dft_input helper if available,
        # but here we construct a dictionary for ase.io.write or just text)

        # We use a dictionary structure compatible with ase.io.espresso.write_espresso_in
        # But since we return string, we can use ase.io.espresso to write to a StringIO

        input_data = {
            "control": {
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "tstress": True,
                "tprnfor": True,
                "disk_io": "low",  # Cleanup large files
                "pseudo_dir": "./",  # Will be handled by runner
                "outdir": "./",  # Will be handled by runner
                "prefix": "calc",
            },
            "system": {
                "ecutwfc": params.get("ecutwfc", 60.0),  # Default heuristic
                "ecutrho": params.get("ecutrho", 240.0),
                "nosym": True,  # As per SPEC
                "occupations": "smearing",
                "smearing": "mv",
                "degauss": params.get("degauss", 0.02),
            },
            "electrons": {
                "mixing_beta": params.get("mixing_beta", 0.7),
                "electron_maxstep": params.get("electron_maxstep", 100),
                "diagonalization": params.get("diagonalization", "david"),
            },
        }

        # Merge magnetism settings
        if magnetism_settings:
            input_data["system"].update(magnetism_settings["system"])
            # ASE handles starting_magnetization via magmom or initial_magnetic_moments on atoms object
            # But we can also force it in input_data if needed.
            # However, for nspin=2, ASE checks atoms.get_initial_magnetic_moments().

        # Apply other overrides
        if "input_data" in params:
            # Deep merge could be better, but simple update for now
            for section, values in params["input_data"].items():
                if section in input_data:
                    input_data[section].update(values)
                else:
                    input_data[section] = values

        # Use ASE to generate the string

        from ase.io.espresso import write_espresso_in

        # We need to set pseudopotentials on the atoms object? No, write_espresso_in takes pseudopotentials arg.

        # Handle magnetism on atoms object
        if magnetism_settings.get("system", {}).get("nspin") == 2:
            # Set initial moments if not present
            moms = atoms.get_initial_magnetic_moments()
            if np.all(moms == 0):
                new_moms = [0.0] * len(atoms)
                for i, atom in enumerate(atoms):
                    if atom.symbol in MAGNETIC_ELEMENTS:
                        new_moms[i] = 2.0  # Default starting mag
                        # Note: SPEC says "assign initial magnetic moments"
                atoms.set_initial_magnetic_moments(new_moms)

        # Capture output in memory
        from io import StringIO

        s_buffer = StringIO()

        write_espresso_in(
            s_buffer,  # ASE < 3.23 uses positional file arg? No, check signature.
            atoms=atoms,
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            kpts=k_points,
            koffset=(0, 0, 0),
        )
        return s_buffer.getvalue()

    @staticmethod
    def _calculate_kpoints(atoms: Atoms, density: float) -> tuple[int, int, int]:
        """
        Calculates K-points based on inverse cell density.
        Nk = max(1, int(L^-1 / density))
        """
        cell = atoms.get_cell()
        lengths = np.linalg.norm(cell, axis=1)
        kpoints = []
        for l in lengths:
            k = int(np.ceil(1.0 / (l * density)))
            kpoints.append(max(1, k))
        return tuple(kpoints)

    @staticmethod
    def _determine_magnetism(atoms: Atoms) -> dict[str, Any]:
        """
        Checks if magnetic elements are present.
        """
        symbols = set(atoms.get_chemical_symbols())
        if not symbols.isdisjoint(MAGNETIC_ELEMENTS):
            return {
                "system": {
                    "nspin": 2,
                    # starting_magnetization(i) is handled by ASE mapping via atoms.magmom usually
                }
            }
        return {}

    @staticmethod
    def _get_pseudopotentials(atoms: Atoms) -> dict[str, str]:
        """
        Returns the pseudopotential mapping for the atoms.
        """
        unique_species = set(atoms.get_chemical_symbols())
        pseudos = {}
        for s in unique_species:
            if s in SSSP_EFFICIENCY_1_1:
                pseudos[s] = SSSP_EFFICIENCY_1_1[s]
            else:
                # Fallback or error? SPEC doesn't say. Let's assume we need it.
                # For now, generic name to prevent crash if not in list
                pseudos[s] = f"{s}.UPF"
        return pseudos
