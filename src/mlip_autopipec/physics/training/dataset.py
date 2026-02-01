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
        Wraps append_structures and finalize_dataset.
        """
        extxyz_path = self.append_structures(structures, append=append)
        return self.finalize_dataset(extxyz_path, output_path)

    def append_structures(
        self, structures: Iterable[Structure], append: bool = True
    ) -> Path:
        """
        Append structures to the intermediate extxyz file.
        """
        extxyz_path = self.work_dir / "dataset.extxyz"

        # Check if iterable is empty without consuming it if possible, or handle via counter
        def atoms_generator() -> Iterator[Atoms]:
            count = 0
            for s in structures:
                yield self._prepare_atoms(s)
                count += 1
            if count == 0:
                logger.debug("No structures provided to append.")

        # Write to file
        write(extxyz_path, atoms_generator(), format="extxyz", append=append)  # type: ignore[arg-type]

        return extxyz_path

    def finalize_dataset(self, extxyz_path: Path, output_path: Path) -> Path:
        """
        Run pace_collect to convert the accumulated extxyz to pckl.gzip.
        """
        logger.info(f"Finalizing dataset: {output_path}")

        if not extxyz_path.exists() or extxyz_path.stat().st_size == 0:
            logger.warning(f"ExtXYZ file {extxyz_path} is empty or missing.")
            # We cannot run pace_collect on empty file
            return output_path

        cmd = ["pace_collect", str(extxyz_path), "--output", str(output_path)]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Dataset finalization successful")
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
