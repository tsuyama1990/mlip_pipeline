import subprocess
from pathlib import Path

from mlip_autopipec.settings import settings


class QEProcessRunner:
    """
    Runs a Quantum Espresso calculation.

    This class takes a directory containing a QE input file and runs the
    calculation using `subprocess.run`.
    """

    def __init__(self, directory: Path) -> None:
        """
        Initializes the QEProcessRunner.

        Args:
            directory: The directory containing the QE input file.
        """
        self.directory = directory

    def run(self) -> subprocess.CompletedProcess:
        """
        Runs the QE calculation and returns the process object.

        Returns:
            The completed process object.
        """
        command = [
            settings.qe_command,
            "-in",
            str(self.directory / "espresso.pwi"),
        ]
        return subprocess.run(  # noqa: S603
            command, capture_output=True, text=True, check=False
        )
