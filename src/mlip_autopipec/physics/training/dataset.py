import subprocess
from pathlib import Path

import ase.io
from mlip_autopipec.domain_models.structure import Structure


class DatasetManager:
    """
    Manages the conversion of Structure objects to Pacemaker datasets.
    """

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def create_dataset(self, structures: list[Structure], name: str) -> Path:
        """
        Convert a list of Structures to a Pacemaker-compatible dataset (.pckl.gzip).

        Args:
            structures: List of Structure objects.
            name: Name of the dataset (without extension).

        Returns:
            Path to the generated .pckl.gzip file.
        """
        extxyz_path = self.work_dir / f"{name}.extxyz"
        output_path = self.work_dir / f"{name}.pckl.gzip"

        # Convert Structures to ASE Atoms
        atoms_list = []
        for s in structures:
            atoms = s.to_ase()

            # Move forces to arrays for correct extxyz writing
            if "forces" in atoms.info:
                forces = atoms.info.pop("forces")
                # Only add if it matches atom count
                if len(forces) == len(atoms):
                    atoms.new_array("forces", forces) # type: ignore[no-untyped-call]

            atoms_list.append(atoms)

        # Write to extended XYZ format
        ase.io.write(extxyz_path, atoms_list, format="extxyz")

        # Run pace_collect
        # Usage: pace_collect -f file.extxyz -o output.pckl.gzip
        cmd = [
            "pace_collect",
            "-f",
            str(extxyz_path),
            "-o",
            str(output_path),
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            msg = f"Dataset conversion failed: {e.stderr}"
            raise RuntimeError(msg) from e
        except FileNotFoundError as e:
            # Handle case where pace_collect is not installed (dev environment)
            msg = "pace_collect executable not found. Ensure pacemaker is installed."
            raise RuntimeError(msg) from e

        if not output_path.exists():
            msg = f"Dataset file was not created at {output_path}"
            raise RuntimeError(msg)

        return output_path
