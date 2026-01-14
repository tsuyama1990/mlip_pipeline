"""Module for managing filesystem operations for Quantum Espresso calculations."""

import tempfile
from pathlib import Path


class QEFileManager:
    """Manages filesystem operations for a Quantum Espresso calculation."""

    def __init__(self) -> None:
        """Initialize the QEFileManager."""
        self._temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._temp_dir.name)

    @property
    def input_path(self) -> Path:
        """Path to the input file."""
        return self.work_dir / "dft.in"

    @property
    def output_path(self) -> Path:
        """Path to the output file."""
        return self.work_dir / "dft.out"

    def write_input(self, content: str) -> None:
        """Write the input file."""
        self.input_path.write_text(content)

    def cleanup(self) -> None:
        """Clean up the temporary directory."""
        self._temp_dir.cleanup()
