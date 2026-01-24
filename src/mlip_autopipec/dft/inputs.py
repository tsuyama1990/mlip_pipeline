from io import StringIO
from typing import Any

import numpy as np
from ase import Atoms
from ase.io.espresso import write_espresso_in

from mlip_autopipec.data_models.dft_models import DFTInputParams
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
    def create_input_string(
        atoms: Atoms, params: DFTInputParams | dict[str, Any] | None = None
    ) -> str:
        """
        Generates the content of a pw.in file.

        Args:
            atoms: The ASE Atoms object.
            params: Parameters for generation. Can be a DFTInputParams model or a dict (for backward compat/tests).
        """
        if params is None:
            params_obj = DFTInputParams()
        elif isinstance(params, dict):
            params_obj = DFTInputParams(**params)
        else:
            params_obj = params

        # 1. Determine K-points
        kspacing = params_obj.kspacing or params_obj.k_density or DEFAULT_KPOINT_DENSITY
        k_points = InputGenerator._calculate_kpoints(atoms, kspacing)

        # 2. Determine Magnetism
        magnetism_settings = InputGenerator._determine_magnetism(atoms)

        # 3. Determine Pseudopotentials
        pseudopotentials = InputGenerator._get_pseudopotentials(atoms)

        # 4. Construct input data
        # Base settings
        input_data = {
            "control": {
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "tstress": True,
                "tprnfor": True,
                "disk_io": "low",
                "pseudo_dir": "./",
                "outdir": "./",
                "prefix": "calc",
            },
            "system": {
                "ecutwfc": params_obj.ecutwfc,
                "ecutrho": params_obj.ecutrho if params_obj.ecutrho else params_obj.ecutwfc * 4,
                "nosym": True,
                "occupations": "smearing",
                "smearing": params_obj.smearing,
                "degauss": params_obj.degauss,
            },
            "electrons": {
                "mixing_beta": params_obj.mixing_beta,
                "electron_maxstep": params_obj.electron_maxstep,
                "diagonalization": params_obj.diagonalization,
            },
        }

        # Merge magnetism settings
        if magnetism_settings:
            input_data["system"].update(magnetism_settings["system"])

        # Apply overrides from input_data field
        if params_obj.input_data:
            for section, values in params_obj.input_data.items():
                if section in input_data:
                    input_data[section].update(values)
                else:
                    input_data[section] = values

        # Enforce Mandatory Flags
        input_data["control"]["tstress"] = True
        input_data["control"]["tprnfor"] = True
        input_data["control"]["disk_io"] = "low"
        input_data["system"]["nosym"] = True

        # Handle magnetism on atoms object
        if input_data["system"].get("nspin") == 2:
            moms = atoms.get_initial_magnetic_moments()
            if np.all(moms == 0):
                new_moms = [0.0] * len(atoms)
                for i, atom in enumerate(atoms):
                    if atom.symbol in MAGNETIC_ELEMENTS:
                        new_moms[i] = 2.0
                atoms.set_initial_magnetic_moments(new_moms)

        # Capture output
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
        if np.all(cell == 0) or not np.any(atoms.pbc):
            return (1, 1, 1)

        lengths = np.linalg.norm(cell, axis=1)
        kpoints = []
        for l in lengths:
            if l < 1e-6:
                k = 1
            else:
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
            return {"system": {"nspin": 2, "starting_magnetization": {}}}
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
