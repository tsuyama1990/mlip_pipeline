import logging
from pathlib import Path

from ase import Atoms
from ase.io import read

from mlip_autopipec.config.schemas.inference import InferenceConfig

log = logging.getLogger(__name__)


class UncertaintyChecker:
    """
    Analyzes LAMMPS dump files to identify and extract high-uncertainty configurations.
    """

    def __init__(self, config: InferenceConfig | None):
        self.config = config
        self.max_gamma = 0.0

    def parse_dump(self, dump_file: Path) -> list[Atoms]:
        """
        Parses the dump file and returns a list of Atoms objects that were flagged as uncertain.

        Args:
            dump_file: Path to the LAMMPS dump file.

        Returns:
            List of ASE Atoms objects.
        """
        if not dump_file.exists() or dump_file.stat().st_size == 0:
            return []

        try:
            # ASE read with index=':' reads all frames
            frames = read(dump_file, index=":", format="lammps-dump-text")

            # Ensure it is a list
            if isinstance(frames, Atoms):
                frames = [frames]

            extracted_frames = []
            max_g = 0.0

            for atoms in frames:
                # Extract timestep from somewhere if possible, usually 'info' or assume sequential
                # ASE lammps-dump-text reader might not preserve timestep in info['src_md_step'] automatically
                # unless mapped. But usually it handles basic properties.
                # If we used 'dump_modify ... thresh', only bad frames are here.

                # Check for gamma in arrays
                # c_pace[1] usually mapped to something like 'c_pace_1_' or 'c_pace[1]'
                gamma_keys = [k for k in atoms.arrays.keys() if "pace" in k or "gamma" in k]
                if gamma_keys:
                    g_vals = atoms.arrays[gamma_keys[0]]
                    current_max = g_vals.max()
                    max_g = max(max_g, current_max)

                    # Store max gamma in info for easier access
                    atoms.info["max_gamma"] = current_max
                    # Assuming the dump only contains frames > threshold, we keep all.
                    # But if we want to double check:
                    if self.config and current_max > self.config.uq_threshold:
                        extracted_frames.append(atoms)
                    elif not self.config:
                        extracted_frames.append(atoms)  # For testing without config

            self.max_gamma = max_g

            # Hack: ASE might not parse timestep into info['src_md_step'] by default for all versions
            # We will rely on what we get.

            return extracted_frames

        except Exception as e:
            log.warning(f"Failed to parse dump file {dump_file}: {e}")
            return []
