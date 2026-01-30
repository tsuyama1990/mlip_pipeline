from typing import Any

import numpy as np
import ase.io
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.calculation import DFTConfig


class InputGenerator:
    """
    Generates input files for Quantum Espresso (pw.x).
    """

    def __init__(self, config: DFTConfig):
        self.config = config

    def generate(self, structure: Structure) -> str:
        """
        Generate the content of a pw.in file.
        """
        atoms = structure.to_ase()

        # Calculate K-points
        kpts = self._calculate_kpoints(atoms)

        # Prepare input_data for ASE
        input_data: dict[str, Any] = {
            "control": {
                "calculation": "scf",
                "tprnfor": self.config.tprnfor,
                "tstress": self.config.tstress,
                "disk_io": "none", # minimize I/O
            },
            "system": {
                "ecutwfc": self.config.ecutwfc,
                "occupations": "smearing",
                "smearing": self.config.smearing,
                "degauss": self.config.degauss,
                "ibrav": 0,
            },
            "electrons": {
                "mixing_beta": self.config.mixing_beta,
                "diagonalization": self.config.diagonalization,
            }
        }

        # Handle pseudopotentials mapping
        # ASE expects pseudopotentials dict: {'Si': 'Si.upf'}
        # Config has {'Si': Path('Si.upf')}
        pseudos = {k: str(v) for k, v in self.config.pseudopotentials.items()}

        # Use ASE to write to a string buffer
        # We need to capture the output. ASE write takes a file object or filename.
        from io import StringIO
        buf = StringIO()

        ase.io.write(
            buf,
            atoms,
            format="espresso-in",
            input_data=input_data,
            pseudopotentials=pseudos,
            kpts=kpts,
            koffset=(1, 1, 1) if self._is_gamma_only(kpts) else (0, 0, 0)
        )

        return buf.getvalue()

    def _calculate_kpoints(self, atoms: ase.Atoms) -> tuple[int, int, int]:
        """
        Calculate K-point grid based on kspacing and cell dimensions.
        N_i = ceil(2 * pi / (L_i * kspacing))
        """
        # Get reciprocal lattice vectors lengths?
        # A better formula is using the reciprocal lattice vectors directly.
        # b_i = 2 * pi / L_i (for orthorhombic)
        # generic: b_i = |b_vec_i|
        # num_k_i = ceil( |b_vec_i| / kspacing )

        # Use atoms.cell.reciprocal() as per deprecation warning
        # ASE's Cell.reciprocal() returns the reciprocal lattice vectors *without* the 2*pi factor?
        # Let's check documentation or assumption.
        # Usually reciprocal lattice b_i * a_j = 2pi * delta_ij in physics (QE uses this).
        # ASE's get_reciprocal_cell() returned 2pi * inverse.
        # atoms.cell.reciprocal() returns Cell object.
        # We need to be careful about the 2pi factor.

        # Let's check if we can rely on standard inverse.
        # The distance in reciprocal space (kspacing) is usually in units of 2pi/A or just 1/A.
        # QE kspacing input is in 1/A (inverse Angstrom) usually? No, QE doesn't take kspacing directly in input,
        # but we are converting kspacing to Grid.
        # If kspacing is 0.04 (1/A), then Grid = RecipLength / 0.04.
        # If RecipLength is 2pi/L, then Grid = 2pi / (L * 0.04).

        # ASE's get_reciprocal_cell() documentation: "Return the reciprocal unit cell ... multiplied by 2*pi."
        # So my previous logic was correct for get_reciprocal_cell().
        # atoms.cell.reciprocal() returns a Cell object representing reciprocal cell.
        # Docs say: "The reciprocal cell is defined as ... a_i . b_j = 2pi delta_ij".
        # So atoms.cell.reciprocal() vectors should have length 2pi/L.

        recip_cell = atoms.cell.reciprocal()
        kpoints = []
        for i in range(3):
            # atoms.cell.reciprocal() returns crystallographic reciprocal vectors (a* . a = 1), so lengths are 1/L.
            # ASE's get_reciprocal_cell() returns 2pi/L, but is deprecated or less preferred for Cell object usage.
            # Physical reciprocal space vectors are b = 2pi * a*.
            # We need physics definition (with 2pi) for kspacing calculation (which is in 1/A).
            # Grid N = |b_phys| / kspacing = (2pi * |a*|) / kspacing.
            b_len = np.linalg.norm(recip_cell[i]) * 2 * np.pi
            n_k = int(np.ceil(b_len / self.config.kspacing))
            kpoints.append(max(1, n_k))

        return tuple(kpoints) # type: ignore[return-value]

    def _is_gamma_only(self, kpts: tuple[int, int, int]) -> bool:
        return kpts == (1, 1, 1)
