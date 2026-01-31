import subprocess
from collections.abc import Iterable
from pathlib import Path

import ase.io
from mlip_autopipec.domain_models.structure import Structure


class DatasetManager:
    """
    Manages the conversion of Structure objects to Pacemaker-compatible datasets.
    """

    def atoms_to_dataset(
        self, structures: Iterable[Structure], output_path: Path
    ) -> Path:
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
        # We process them one by one to avoid huge memory spike if iterable is lazy
        # However, ase.io.write handles list better for 'extxyz' format sometimes.
        # But 'extxyz' supports append.

        # Clean existing file if any
        if extxyz_path.exists():
            extxyz_path.unlink()

        atoms_list = []
        for s in structures:
            atoms = s.to_ase()
            # Ensure properties are in info/arrays for extxyz to pick them up correctly
            # ASE's extxyz writer looks at atoms.info for scalars and atoms.arrays for per-atom
            # Structure.to_ase puts everything in atoms.info
            # We might need to move 'forces' to arrays if it's in info?
            # Actually, ASE Atoms 'forces' are usually handled by Calculator or get_forces().
            # But here we are just data containers.
            # extxyz writer expects 'forces' in arrays (arrays['forces']) or calc.

            # Let's check how Structure.to_ase handles it.
            # Structure puts everything in info.
            # We need to move vector properties to arrays or set them via set_...

            # Standard ASE keys: energy, forces, stress
            props = s.properties
            if "energy" in props:
                atoms.info["energy"] = props["energy"]
            if "forces" in props:
                # forces is (N, 3), should be in arrays or attached as calculator
                # Setting it in arrays explicitly for writing
                atoms.new_array("forces", props["forces"])  # type: ignore[no-untyped-call]
            if "stress" in props:
                atoms.info["stress"] = props["stress"]
            if "virial" in props:
                atoms.info["virial"] = props["virial"]

            atoms_list.append(atoms)

        # Write to extxyz
        ase.io.write(extxyz_path, atoms_list, format="extxyz")

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
