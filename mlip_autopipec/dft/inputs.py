from typing import Any

import numpy as np
from ase import Atoms
from ase.io.espresso import write_espresso_in
from io import StringIO

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
            params: Optional overrides for parameters. Dictionary is used here for flexibility
                    but should ideally align with DFTInputParameters model where possible.
        """
        if params is None:
            params = {}

        # 1. Determine K-points based on kspacing
        # Prefer 'kspacing' key, fallback to 'k_density' for compat, default to constant
        kspacing = params.get("kspacing", params.get("k_density", DEFAULT_KPOINT_DENSITY))
        k_points = InputGenerator._calculate_kpoints(atoms, kspacing)

        # 2. Determine Magnetism
        magnetism_settings = InputGenerator._determine_magnetism(atoms)

        # 3. Determine Pseudopotentials
        pseudopotentials = InputGenerator._get_pseudopotentials(atoms)

        # 4. Construct the input data
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
                "ecutrho": params.get("ecutrho", params.get("ecutwfc", 60.0) * 4), # ecutrho usually 4*ecutwfc
                "nosym": True,  # As per SPEC
                "occupations": "smearing",
                "smearing": params.get("smearing", "mv"),
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

        # Apply other overrides from params['input_data']
        if "input_data" in params:
            for section, values in params["input_data"].items():
                if section in input_data:
                    input_data[section].update(values)
                else:
                    input_data[section] = values

        # Enforce Mandatory Flags (Safety & Training Requirements)
        input_data["control"]["tstress"] = True
        input_data["control"]["tprnfor"] = True
        input_data["control"]["disk_io"] = "low"
        input_data["system"]["nosym"] = True

        # Handle magnetism on atoms object (ASE < 3.23 compat)
        if input_data["system"].get("nspin") == 2:
            # Set initial moments if not present on atoms
            # Check existing moments
            moms = atoms.get_initial_magnetic_moments()
            # If all zeros, set default for magnetic elements
            if np.all(moms == 0):
                new_moms = [0.0] * len(atoms)
                for i, atom in enumerate(atoms):
                    if atom.symbol in MAGNETIC_ELEMENTS:
                        new_moms[i] = 2.0  # Default starting mag
                atoms.set_initial_magnetic_moments(new_moms)

        # Capture output in memory
        s_buffer = StringIO()

        write_espresso_in(
            s_buffer,
            atoms=atoms,
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            kpts=k_points,
            koffset=(0, 0, 0),
        )
        return s_buffer.getvalue()

    @staticmethod
    def _calculate_kpoints(atoms: Atoms, kspacing: float) -> tuple[int, int, int]:
        """
        Calculates K-points based on kspacing (Inverse K-point density).
        Nk = ceil(2 * pi / (L * kspacing))
        """
        cell = atoms.get_cell()
        # Handle non-periodic or zero cell
        if np.all(cell == 0) or not np.any(atoms.pbc):
             return (1, 1, 1)

        lengths = np.linalg.norm(cell, axis=1)
        kpoints = []
        for l in lengths:
            if l < 1e-6: # Avoid division by zero for collapsed cells
                k = 1
            else:
                # Reciprocal lattice vector length b ~ 2pi / a
                # Nk > 2pi / (a * kspacing)
                k = int(np.ceil(2 * np.pi / (l * kspacing)))
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
                    "starting_magnetization": {} # Placeholder, ASE handles it via magmom
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
                pseudos[s] = f"{s}.UPF"
        return pseudos
