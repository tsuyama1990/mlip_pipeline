import logging
import subprocess
from pathlib import Path
from typing import List, Iterable, Union

import ase.io
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class DatasetManager:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def convert(self, structures: Iterable[Structure], output_path: Path) -> Path:
        """
        Converts an iterable of Structures to a pacemaker-compatible dataset.

        Args:
            structures: Iterable of Structure objects.
            output_path: Path to save the .pckl.gzip dataset.

        Returns:
            The path to the saved dataset.
        """
        temp_extxyz = self.work_dir / "temp_dataset.extxyz"

        # Generator that yields ASE Atoms
        def ase_generator():
            for s in structures:
                atoms = s.to_ase()

                # Move physical properties from info to Calculator for valid extxyz format
                properties = atoms.info
                results = {}
                if "energy" in properties:
                    results["energy"] = properties.pop("energy")

                if "forces" in properties:
                    results["forces"] = properties.pop("forces")

                if "stress" in properties:
                    results["stress"] = properties.pop("stress")

                if "magmoms" in properties:
                    results["magmoms"] = properties.pop("magmoms")

                if results:
                    calc = SinglePointCalculator(atoms, **results)
                    atoms.calc = calc

                yield atoms

        # Write to extxyz using the generator to stream data
        logger.info(f"Streaming structures to {temp_extxyz}")

        # ase.io.write supports iterables of Atoms
        ase.io.write(temp_extxyz, ase_generator(), format="extxyz")

        # Call pace_collect
        # pace_collect <input_extxyz> <output_pckl>
        logger.info(f"Converting to {output_path} using pace_collect")
        cmd = ["pace_collect", str(temp_extxyz), str(output_path)]

        try:
            # check=True raises CalledProcessError on non-zero exit
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"pace_collect failed: {e.stdout} \n {e.stderr}")
            raise RuntimeError(f"pace_collect failed: {e.stderr}") from e

        return output_path
