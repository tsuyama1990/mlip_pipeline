from io import StringIO
from typing import Any

import numpy as np
from ase.io import write # type: ignore

from mlip_autopipec.domain_models.calculation import DFTConfig
from mlip_autopipec.domain_models.structure import Structure


class InputGenerator:
    """Generates Quantum Espresso input files."""

    @staticmethod
    def generate(structure: Structure, config: DFTConfig) -> str:
        """
        Generate the content of a pw.x input file.

        Args:
            structure: The atomic structure.
            config: DFT configuration parameters.

        Returns:
            The input file content as a string.
        """
        atoms = structure.to_ase()

        # Calculate K-points based on kspacing
        # Formula: Ni = ceil(2 * pi / (|a_i| * kspacing))
        # We use the lengths of the cell vectors.
        cell_lengths = np.linalg.norm(structure.cell, axis=1)

        # Avoid division by zero if cell is effectively zero (shouldn't happen for valid Structure)
        # But cell lengths could be small.
        kpoints = np.ceil(2 * np.pi / (cell_lengths * config.kspacing)).astype(int)
        kpoints = np.maximum(kpoints, 1)  # Ensure at least 1 k-point

        # Prepare input_data for ASE
        input_data: dict[str, Any] = {
            'control': {
                'calculation': 'scf',
                'restart_mode': 'from_scratch',
                'pseudo_dir': '.',  # Runner handles file placement
                'outdir': './out',
                'tprnfor': True,
                'tstress': True,
                'disk_io': 'low',
            },
            'system': {
                'ecutwfc': config.ecutwfc,
                'smearing': config.smearing,
                'degauss': config.degauss,
                'occupations': 'smearing',
                'nat': len(atoms),
                'ntyp': len(set(atoms.get_chemical_symbols())), # type: ignore[no-untyped-call]
            },
            'electrons': {
                'mixing_beta': config.mixing_beta,
                'diagonalization': config.diagonalization,
                'conv_thr': 1.0e-6,
            }
        }

        # Pseudopotentials mapping
        # ASE expects {symbol: filename}
        pseudopotentials = {
            el: str(path.name) for el, path in config.pseudopotentials.items()
        }

        # Generate string
        f = StringIO()
        write(
            f,
            atoms,
            format='espresso-in',
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            kpts=tuple(kpoints),
            koffset=(0, 0, 0) # Gamma centered
        )

        return f.getvalue()
