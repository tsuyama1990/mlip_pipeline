"""
Input Writer Module.

This module provides functionality to write LAMMPS input files (data and script).
"""

from pathlib import Path

from ase.atoms import Atoms
from ase.io import write

from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.inference.inputs import ScriptGenerator


class LammpsInputWriter:
    """Handles writing of LAMMPS input files."""

    def __init__(self, config: InferenceConfig, work_dir: Path) -> None:
        """
        Initialize the writer.

        Args:
            config: Inference configuration.
            work_dir: Working directory for files.
        """
        self.config = config
        self.work_dir = work_dir
        self.generator = ScriptGenerator(config)

    def write_inputs(self, atoms: Atoms, potential_path: Path) -> tuple[Path, Path, Path, Path]:
        """
        Writes data file and input script to disk.

        Args:
            atoms: Atomic structure.
            potential_path: Path to the potential file (.yace).

        Returns:
            Tuple of (input_file, data_file, log_file, dump_file) paths.
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)

        input_file = self.work_dir / "in.lammps"
        data_file = self.work_dir / "data.lammps"
        log_file = self.work_dir / "log.lammps"
        dump_file = self.work_dir / "dump.gamma"

        # Determine elements order (sorted for consistency)
        elements = sorted(set(atoms.get_chemical_symbols()))

        # Write Data File with specific element order
        write(data_file, atoms, format="lammps-data", specorder=elements)

        # Generate Input Script
        script_content = self.generator.generate(
            atoms_file=data_file,
            potential_path=potential_path,
            dump_file=dump_file,
            elements=elements
        )

        input_file.write_text(script_content)

        return input_file, data_file, log_file, dump_file
