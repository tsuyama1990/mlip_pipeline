from math import ceil, pi
from pathlib import Path

import numpy as np

from mlip_autopipec.domain_models.calculation import DFTConfig
from mlip_autopipec.domain_models.structure import Structure


class InputGenerator:
    """Generates Quantum Espresso input files."""

    def generate_input(self, structure: Structure, config: DFTConfig) -> str:
        """
        Generate the content of a pw.x input file.
        """
        # 1. System Block
        # We need to map species to types.
        # Structure symbols: ["Si", "Si"] -> ntyp=1
        unique_species = sorted(list(set(structure.symbols)))
        ntyp = len(unique_species)
        nat = len(structure.symbols)

        # Calculate cell parameters
        # For simplicity, we write CELL_PARAMETERS card (Angstrom)

        # 2. K-Points
        # Formula: Ni = ceil(2 * pi / (|a_i| * kspacing))
        # This assumes orthogonal or simply using vector lengths.
        # For non-orthogonal, it's better to use reciprocal lattice vectors magnitude.
        # |b_i| / kspacing

        # Calculate reciprocal lattice
        # cell is (3,3) rows are a1, a2, a3.
        # Reciprocal vectors b1 = 2pi * (a2 x a3) / (a1 . (a2 x a3))
        # Or simply: B = 2pi * inv(A).T

        cell = structure.cell
        # Check for zero volume? Structure model validation ensures valid cell?
        # Maybe not. But let's assume valid.

        if np.abs(np.linalg.det(cell)) < 1e-8:
             # Fallback for degenerate cell (shouldn't happen in DFT)
             recip_cell = np.eye(3)
        else:
             recip_cell = 2 * pi * np.linalg.inv(cell).T

        b_norms = np.linalg.norm(recip_cell, axis=1)

        k_grid = [
            max(1, int(ceil(b_norm / config.kspacing)))
            for b_norm in b_norms
        ]

        # 3. Construct String
        lines = []

        # CONTROL
        lines.append("&CONTROL")
        lines.append("  calculation = 'scf'")
        lines.append("  pseudo_dir = '.'") # Will be handled by runner putting pseudos in dir
        lines.append("  outdir = './tmp'")
        lines.append("  tprnfor = .true.")
        lines.append("  tstress = .true.")
        lines.append("  disk_io = 'low'")
        lines.append("/")

        # SYSTEM
        lines.append("&SYSTEM")
        lines.append("  ibrav = 0")
        lines.append(f"  nat = {nat}")
        lines.append(f"  ntyp = {ntyp}")
        lines.append(f"  ecutwfc = {config.ecutwfc}")

        # Add smearing if metal/configured
        if config.smearing:
            lines.append("  occupations = 'smearing'")
            lines.append(f"  smearing = '{config.smearing}'")
            lines.append(f"  degauss = {config.degauss}")

        lines.append("/")

        # ELECTRONS
        lines.append("&ELECTRONS")
        lines.append(f"  mixing_beta = {config.mixing_beta}")
        lines.append(f"  diagonalization = '{config.diagonalization}'")
        lines.append("  conv_thr = 1.0d-8")
        lines.append("/")

        # ATOMIC_SPECIES
        lines.append("ATOMIC_SPECIES")
        for species in unique_species:
            # We assume mass is auto-read from UPF, so assume 1.0 or lookup.
            # QE reads mass from UPF if set to something.
            # Need atomic mass?
            # If we don't provide mass, QE uses default or UPF?
            # Usually: "Si  28.086  Si.upf"
            # We can use ASE's atomic_masses
            from ase.data import atomic_masses, atomic_numbers
            mass = atomic_masses[atomic_numbers[species]]
            pseudo_file = config.pseudopotentials.get(species, f"{species}.upf")
            if isinstance(pseudo_file, Path):
                pseudo_file = pseudo_file.name
            lines.append(f"  {species} {mass:.4f} {pseudo_file}")

        # ATOMIC_POSITIONS
        lines.append("ATOMIC_POSITIONS {angstrom}")
        for s, p in zip(structure.symbols, structure.positions):
            lines.append(f"  {s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}")

        # K_POINTS
        lines.append("K_POINTS automatic")
        lines.append(f"  {k_grid[0]} {k_grid[1]} {k_grid[2]} 0 0 0")

        # CELL_PARAMETERS
        lines.append("CELL_PARAMETERS {angstrom}")
        for row in cell:
            lines.append(f"  {row[0]:.8f} {row[1]:.8f} {row[2]:.8f}")

        return "\n".join(lines)
