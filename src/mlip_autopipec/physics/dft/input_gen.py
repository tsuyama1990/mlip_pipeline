from io import StringIO
import numpy as np
from ase.io.espresso import write_espresso_in

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.calculation import DFTConfig


class InputGenerator:
    def generate_input(self, structure: Structure, config: DFTConfig) -> str:
        atoms = structure.to_ase()

        # Calculate K-points
        # ASE cell.reciprocal() returns crystallographic vectors (no 2pi)
        recip_cell = atoms.cell.reciprocal()  # type: ignore[no-untyped-call]
        recip_lengths = np.linalg.norm(recip_cell, axis=1)
        kpts = np.ceil(2 * np.pi * recip_lengths / config.kspacing).astype(int)

        # Ensure at least 1x1x1
        kpts = np.maximum(kpts, 1)

        # Convert to tuple so ASE treats it as MP grid
        kpts_tuple = tuple(kpts.tolist())

        input_data = {
            "control": {
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "tprnfor": True,
                "tstress": True,
                "disk_io": "none",
                "pseudo_dir": ".",
                "outdir": "./out",
            },
            "system": {
                "ecutwfc": config.ecutwfc,
                "occupations": "smearing",
                "smearing": config.smearing,
                "degauss": config.degauss,
                "ibrav": 0,  # Explicit lattice
            },
            "electrons": {
                "mixing_beta": config.mixing_beta,
                "conv_thr": 1.0e-6,
            },
        }

        # Pseudopotentials mapping
        pseudos = {el: path.name for el, path in config.pseudopotentials.items()}

        f = StringIO()
        write_espresso_in(
            f,
            atoms,
            input_data=input_data,
            pseudopotentials=pseudos,
            kpts=kpts_tuple,
            koffset=(0, 0, 0),
        )
        return f.getvalue()
