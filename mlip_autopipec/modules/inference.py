"""
This module contains the `LammpsRunner` class and the core logic for
On-The-Fly (OTF) inference, including uncertainty detection and the
Periodic Embedding/Force Masking data extraction strategies.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.geometry import find_mic, wrap_positions
from ase.io import read, write

from mlip_autopipec.config.models import (
    InferenceConfig,
    UncertainStructure,
    UncertaintyConfig,
    UncertaintyMetadata,
)

log = logging.getLogger(__name__)


def extract_embedded_structure(
    large_cell: Atoms, center_atom_index: int, config: UncertaintyConfig
) -> UncertainStructure:
    """
    Extracts a small, periodic sub-system centered on an atom of interest.

    This function performs two key operations:
    1.  Periodic Embedding: It carves out a smaller, fully periodic `ase.Atoms`
        object from a larger simulation cell, correctly wrapping atoms across
        periodic boundaries.
    2.  Force Masking: It generates a boolean mask to distinguish "core" atoms
        (whose forces are reliable for training) from "buffer" atoms at the
        edge of the new cell.

    Args:
        large_cell: The full simulation cell from which to extract.
        center_atom_index: The index of the uncertain atom.
        config: The uncertainty configuration containing cutoff radii.

    Returns:
        An `UncertainStructure` object containing the new `ase.Atoms` object
        and the corresponding force mask.
    """
    # 1. Periodic Embedding
    cutoff = config.embedding_cutoff
    center_pos = large_cell.positions[center_atom_index]

    # Create a new cell matrix for the embedded structure
    new_cell = np.diag([2 * cutoff, 2 * cutoff, 2 * cutoff])

    # Wrap all positions relative to the center atom to handle PBC
    wrapped_positions = wrap_positions(
        large_cell.positions, large_cell.cell, pbc=large_cell.pbc, center=center_pos
    )

    # Find atoms within the spherical cutoff of the *unwrapped* center position
    distances = np.linalg.norm(wrapped_positions - center_pos, axis=1)
    indices_in_cutoff = np.where(distances < cutoff)[0]

    # Create the new Atoms object with the selected atoms
    embedded_atoms = large_cell[indices_in_cutoff].copy()
    embedded_atoms.set_cell(new_cell)
    embedded_atoms.set_pbc(True)

    # Center the atoms in the new cell
    embedded_atoms.center()

    # 2. Force Masking
    masking_cutoff = config.masking_cutoff
    center_of_new_cell = embedded_atoms.get_center_of_mass()

    # Manually calculate MIC distances to avoid ASE's quirky API
    _, distances_from_center = find_mic(
        embedded_atoms.positions - center_of_new_cell, embedded_atoms.cell, pbc=True
    )
    force_mask = (distances_from_center < masking_cutoff).astype(int)

    metadata = UncertaintyMetadata(
        uncertain_timestep=0,
        uncertain_atom_id=0,
        uncertain_atom_index_in_original_cell=center_atom_index,
    )
    return UncertainStructure(
        atoms=embedded_atoms, force_mask=force_mask, metadata=metadata
    )


class LammpsRunner:
    """
    A component for running molecular dynamics (MD) simulations with LAMMPS to
    drive active learning.

    The `LammpsRunner` takes a trained MLIP and an initial atomic structure, then
    runs an MD simulation. It continuously monitors the uncertainty of the MLIP's
    predictions for each atom. If any atom's uncertainty exceeds a predefined
    threshold, the simulation is stopped.

    The primary output of this process is an `UncertainStructure` object, which
    contains the atomic configuration that triggered the high uncertainty, along
    with a force mask. This object is specifically designed to be passed to the
    DFTFactory for high-fidelity re-calculation, closing the active learning loop.
    """

    def __init__(self, inference_config: InferenceConfig) -> None:
        """
        Initializes the LammpsRunner.

        Args:
            inference_config: A Pydantic model containing all necessary paths
                              and parameters for the simulation and uncertainty
                              detection.
        """
        self.config = inference_config

    def run(self, initial_structure: Atoms) -> Optional[UncertainStructure]:
        """
        Runs a LAMMPS MD simulation and monitors it for uncertainty.

        If the simulation completes without any atom's uncertainty exceeding
        the configured threshold, it returns `None`.

        If high uncertainty is detected, the simulation is stopped, and this
        method returns an `UncertainStructure` object containing the extracted,
        embedded structure and its force mask, ready for DFT calculation.

        Args:
            initial_structure: The starting `ase.Atoms` object for the MD run.

        Returns:
            An `UncertainStructure` object if the uncertainty threshold was
            exceeded, otherwise `None`.
        """
        with tempfile.TemporaryDirectory() as temp_dir_str:
            working_dir = Path(temp_dir_str)
            structure_file = working_dir / "structure.data"
            write(structure_file, initial_structure, format="lammps-data")

            input_script = self._prepare_lammps_input(
                initial_structure,
                working_dir,
                structure_file,
                self.config.potential_path,
            )

            log.info("Starting LAMMPS simulation...")
            self._execute_lammps(working_dir, input_script)
            log.info("LAMMPS simulation finished.")

            uncertain_frame_info = self._find_first_uncertain_frame(working_dir)

            if uncertain_frame_info:
                timestep, atom_id = uncertain_frame_info
                log.info(
                    f"Uncertainty threshold exceeded at timestep {timestep} "
                    f"by atom ID {atom_id}."
                )
                frame_index = self._get_frame_index_for_timestep(
                    working_dir / "dump.custom", timestep
                )
                if frame_index is None:
                    log.error(f"Could not find timestep {timestep} in trajectory file.")
                    return None

                full_structure = read(
                    working_dir / "dump.custom",
                    index=frame_index,
                    format="lammps-dump-text",
                )
                atom_index_in_cell = atom_id - 1

                extracted_data = extract_embedded_structure(
                    full_structure,
                    atom_index_in_cell,
                    self.config.uncertainty_params,
                )
                extracted_data.metadata.uncertain_timestep = timestep
                extracted_data.metadata.uncertain_atom_id = atom_id
                return extracted_data

            log.info("No uncertainty detected within the simulation run.")
            return None

    def _prepare_lammps_input(
        self,
        atoms: Atoms,
        working_dir: Path,
        structure_file: Path,
        potential_file: Path,
    ) -> Path:
        """Generates the LAMMPS input script."""
        symbols = sorted(set(atoms.get_chemical_symbols()))

        script = f"""
        units           metal
        boundary        p p p
        atom_style      atomic

        read_data       {structure_file.name}

        pair_style      pace
        pair_coeff      * * {potential_file.name} {" ".join(symbols)}

        thermo          100
        thermo_style    custom step temp pe etotal press

        compute         uncert all pace/extrapol
        dump            uncert_dump all custom 10 uncertainty.dump c_uncert[*]

        dump            traj_dump all custom 10 dump.custom id type x y z

        timestep        {self.config.md_params.timestep}
        fix             1 all {self.config.md_params.ensemble} temp {self.config.md_params.temperature} {self.config.md_params.temperature} 0.1

        run             {self.config.md_params.run_duration}
        """
        script_path = working_dir / "in.lammps"
        script_path.write_text(script)
        return script_path

    def _execute_lammps(
        self, working_dir: Path, input_script: Path
    ) -> Optional[subprocess.CompletedProcess[str]]:
        """Executes the LAMMPS simulation as a subprocess."""
        cmd = [str(self.config.lammps_executable), "-in", str(input_script)]
        log.debug(f"Running LAMMPS command: {' '.join(cmd)}")
        try:
            return subprocess.run(
                cmd,
                cwd=str(working_dir),
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError:
            log.exception(
                f"Lammps executable not found at {self.config.lammps_executable}"
            )
            raise
        except subprocess.CalledProcessError as e:
            log.exception(f"LAMMPS simulation failed with exit code {e.returncode}")
            raise

    def _get_frame_index_for_timestep(
        self, trajectory_file: Path, target_timestep: int
    ) -> Optional[int]:
        """Finds the 0-based index of a frame for a given timestep in a dump file."""
        if not trajectory_file.exists():
            return None
        with trajectory_file.open() as f:
            lines = f.readlines()

        frame_index = 0
        for i, line in enumerate(lines):
            if "ITEM: TIMESTEP" in line:
                timestep_on_line = int(lines[i + 1])
                if timestep_on_line == target_timestep:
                    return frame_index
                frame_index += 1
        return None

    def _parse_dump_file(self, file_path: Path) -> Dict[int, np.ndarray]:
        """Parses a LAMMPS dump file and returns a dictionary mapping timestep to data."""
        timesteps: Dict[int, np.ndarray] = {}
        if not file_path.exists():
            return timesteps

        with file_path.open() as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            if "ITEM: TIMESTEP" in line:
                current_timestep = int(lines[i + 1])
                i += 2
            elif "ITEM: NUMBER OF ATOMS" in line:
                num_atoms = int(lines[i + 1])
                i += 2
            elif "ITEM: ATOMS" in line:
                data_lines = lines[i + 1 : i + 1 + num_atoms]
                if data_lines:
                    data = np.loadtxt(data_lines)
                    timesteps[current_timestep] = data
                i += num_atoms + 1
            else:
                i += 1
        return timesteps

    def _find_first_uncertain_frame(
        self, working_dir: Path
    ) -> Optional[Tuple[int, int]]:
        """Parses LAMMPS output for the first frame exceeding the uncertainty threshold."""
        uncertainty_data = self._parse_dump_file(working_dir / "uncertainty.dump")

        for timestep, uncert_values in sorted(uncertainty_data.items()):
            if uncert_values.ndim == 0:
                uncert_values_arr = np.array([uncert_values])
            else:
                uncert_values_arr = uncert_values

            if uncert_values_arr.max() > self.config.uncertainty_params.threshold:
                uncertain_atom_index = np.argmax(uncert_values_arr)

                traj_file = working_dir / "dump.custom"
                with traj_file.open() as f:
                    traj_lines = f.readlines()

                atom_id = self._find_atom_id_in_frame(
                    traj_lines,
                    timestep,
                    uncertain_atom_index,
                )
                if atom_id:
                    return timestep, atom_id

        return None

    def _find_atom_id_in_frame(
        self,
        traj_lines: List[str],
        timestep: int,
        atom_index: int,
    ) -> Optional[int]:
        """Finds the ID of a specific atom in a specific frame of a trajectory."""
        for i, line in enumerate(traj_lines):
            if "ITEM: TIMESTEP" in line and int(traj_lines[i + 1]) == timestep:
                j = i
                while "ITEM: ATOMS" not in traj_lines[j]:
                    j += 1
                atom_lines_start = j + 1
                line_with_uncertain_atom = traj_lines[atom_lines_start + atom_index]
                return int(line_with_uncertain_atom.split()[0])
        return None
