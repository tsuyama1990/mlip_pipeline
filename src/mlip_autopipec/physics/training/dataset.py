import subprocess
from collections.abc import Iterable
from pathlib import Path

import ase.io
from mlip_autopipec.domain_models.structure import Structure


class DatasetManager:
    """
    Manages the conversion of Structure objects to Pacemaker-compatible datasets.
    """

    def atoms_to_dataset(self, structures: Iterable[Structure], output_path: Path) -> Path:
        """
        Convert a list of Structure objects to a .pckl.gzip dataset.

        Args:
            structures: List of Structure objects.
            output_path: Destination path for the .pckl.gzip file.

        Returns:
            Path to the generated dataset.
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Intermediate extxyz file
        # We use the same name stem but with .extxyz extension
        extxyz_path = output_path.with_suffix("").with_suffix(".extxyz")

        # Convert structures to ASE atoms
        # Use a generator to stream data and avoid loading everything into memory
        def atoms_generator() -> Iterable[ase.Atoms]:
            for s in structures:
                atoms = s.to_ase()
                # Ensure properties are in info/arrays for extxyz to pick them up correctly

                # Standard ASE keys: energy, forces, stress
                props = s.properties
                if "energy" in props:
                    atoms.info["energy"] = props["energy"]
                if "forces" in props:
                    # forces is (N, 3), should be in arrays or attached as calculator
                    # Setting it in arrays explicitly for writing
                    atoms.new_array("forces", props["forces"]) # type: ignore[no-untyped-call]
                if "stress" in props:
                    atoms.info["stress"] = props["stress"]
                if "virial" in props:
                     atoms.info["virial"] = props["virial"]

                yield atoms

        # Write to extxyz
        # ase.io.write supports iterables for 'extxyz' format, writing frame by frame
        ase.io.write(extxyz_path, atoms_generator(), format="extxyz")  # type: ignore[arg-type]

        # Call pace_collect
        # Command: pace_collect --extxyz <file.extxyz> --output <file.pckl.gzip>
        cmd = [
            "pace_collect",
            "--extxyz",
            str(extxyz_path),
            "--output",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            msg = f"pace_collect failed: {e.stderr}"
            raise RuntimeError(msg) from e

        return output_path
