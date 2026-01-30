from io import StringIO
from typing import Any, Optional

import numpy as np
from ase.io.espresso import write_espresso_in

from mlip_autopipec.domain_models.calculation import DFTConfig
from mlip_autopipec.domain_models.structure import Structure


class InputGenerator:
    """
    Generates input files for DFT calculations (Quantum Espresso).
    """

    @staticmethod
    def calculate_kpoints(cell: np.ndarray, kspacing: float) -> tuple[int, int, int]:
        """
        Calculate the number of K-points along each reciprocal lattice vector
        to satisfy the given k-spacing density (inverse Angstroms).

        N_i = ceil(2 * pi / (L_i * kspacing))
        """
        lengths = np.linalg.norm(cell, axis=1)
        # Avoid division by zero
        lengths = np.where(lengths < 1e-6, 1.0, lengths)

        kpoints = np.ceil(2 * np.pi / (lengths * kspacing)).astype(int)
        # Ensure at least 1
        kpoints = np.maximum(kpoints, 1)

        return tuple(kpoints)  # type: ignore

    @staticmethod
    def generate_input(
        structure: Structure,
        config: DFTConfig,
        parameters: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Generate the content of a Quantum Espresso input file.
        """
        atoms = structure.to_ase()

        # Base parameters required for MLIP training
        input_data: dict[str, dict[str, Any]] = {
            "control": {
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "tprnfor": True,
                "tstress": True,
                "disk_io": "low",  # Avoid huge files
            },
            "system": {
                "ecutwfc": config.ecutwfc,
                "nosym": False,
            },
            "electrons": {
                "mixing_beta": 0.7,
                "conv_thr": 1.0e-6,
            },
        }

        # To support flat overrides easily, let's flatten the default `input_data`
        # and merge with `parameters`.
        flat_input_data: dict[str, Any] = {}
        for section, content in input_data.items():
            flat_input_data.update(content)

        if parameters:
            flat_input_data.update(parameters)

        # Handle pseudopotentials
        # ASE expects pseudopotentials as a dict {Element: Filename}
        # config.pseudopotentials is Dict[str, Path]
        pseudos = {el: str(path) for el, path in config.pseudopotentials.items()}

        # Calculate K-Points
        kpts = InputGenerator.calculate_kpoints(atoms.cell.array, config.kspacing)  # type: ignore

        # Use ASE to write to string
        output = StringIO()
        write_espresso_in(
            output,
            atoms,
            input_data=flat_input_data,
            pseudopotentials=pseudos,
            kpts=kpts,
            koffset=(0, 0, 0),  # Gamma centered usually? Or MP? (0,0,0) is fine for MP if grid is dense.
        )

        return output.getvalue()
