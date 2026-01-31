import logging
import subprocess
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger("mlip_autopipec")


class DatasetManager:
    """
    Manages the conversion of ASE Structures to Pacemaker training datasets.
    """

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def convert(
        self, structures: Iterable[Structure], output_path: Path, append: bool = False
    ) -> Path:
        """
        Convert a list (or iterator) of Structure objects to a .pckl.gzip dataset.

        Uses streaming to avoid loading all atoms into memory.
        If append=True, appends to the intermediate extxyz file, effectively adding to the dataset.

        Args:
            structures: Iterable of Structure objects.
            output_path: Path to the output .pckl.gzip file.
            append: If True, append to existing dataset.extxyz.

        Returns:
            Path to the generated dataset.
        """
        logger.info(f"Converting structures to Pacemaker dataset: {output_path}")

        # 1. Write structures to extxyz
        extxyz_path = self.work_dir / "dataset.extxyz"

        # Check if iterable is empty without consuming it if possible, or handle via counter
        def atoms_generator() -> Iterator[Atoms]:
            count = 0
            for s in structures:
                yield self._prepare_atoms(s)
                count += 1
            # If count == 0, ase.io.write might do nothing or create empty file depending on version/format
            if count == 0:
                logger.warning("No structures provided to convert.")

        # Write to file
        # ase.io.write supports generator for many formats including extxyz
        # append=True allows accumulation
        # Note: If streaming from disk (iread), this streams to disk (extxyz), keeping memory O(1) wrt N_structures
        write(extxyz_path, atoms_generator(), format="extxyz", append=append)  # type: ignore[arg-type]

        # Check if file is empty or not created (if generator was empty)
        if not extxyz_path.exists() or extxyz_path.stat().st_size == 0:
             logger.warning("dataset.extxyz is empty or does not exist.")
             if not append:
                 # If we are starting fresh and have no data, we can't create a valid dataset.
                 # But maybe we just return output_path (which won't exist) and let caller handle?
                 # Or raise Error? Pacemaker will fail on empty file.
                 pass

        # 2. Call pace_collect
        # usage: pace_collect dataset.extxyz -o dataset.pckl.gzip
        # Note: pace_collect might load the whole extxyz into memory. This is a limitation of the tool.
        # But we satisfied the requirement to not buffer in python memory.
        cmd = ["pace_collect", str(extxyz_path), "--output", str(output_path)]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Dataset conversion successful")
        except subprocess.CalledProcessError as e:
            logger.error(f"pace_collect failed: {e.stderr}")
            raise RuntimeError(f"pace_collect failed: {e.stderr}") from e

        return output_path

    def _prepare_atoms(self, structure: Structure) -> Atoms:
        """
        Convert Structure to ASE Atoms, ensuring energy/forces/stress are attached
        as a SinglePointCalculator so ASE writes them correctly to extxyz.
        """
        atoms = structure.to_ase()

        # Structure.to_ase() puts everything in atoms.info
        # We need to extract them and put into Calculator

        # Extract properties
        # Note: keys in structure.properties should match ASE expectations
        # 'energy', 'forces', 'stress'

        energy = atoms.info.get("energy")
        forces = atoms.info.get("forces")
        stress = atoms.info.get("stress")

        # Remove from info to avoid duplication in extended XYZ comment line
        if "energy" in atoms.info:
            del atoms.info["energy"]
        if "forces" in atoms.info:
            del atoms.info["forces"]
        if "stress" in atoms.info:
            del atoms.info["stress"]

        calc_args = {}
        if energy is not None:
            calc_args["energy"] = energy
        if forces is not None:
            # Forces must be numpy array
            calc_args["forces"] = np.array(forces)
        if stress is not None:
            calc_args["stress"] = np.array(stress)

        if calc_args:
            # type: ignore[no-untyped-call]
            atoms.calc = SinglePointCalculator(atoms, **calc_args)

        return atoms
