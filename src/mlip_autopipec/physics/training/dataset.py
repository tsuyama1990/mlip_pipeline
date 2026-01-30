import logging
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Iterator

import ase
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger("mlip_autopipec")


class DatasetManager:
    def convert(self, structures: Iterable[Structure], output_path: Path) -> Path:
        """
        Convert an iterable of Structure objects to a pacemaker-compatible dataset.
        Uses a generator to stream writing to avoid OOM with large datasets.
        """

        def atom_generator() -> Iterator[ase.Atoms]:
            for s in structures:
                atoms = s.to_ase()

                # Extract labels from properties
                energy = s.properties.get("energy")
                forces = s.properties.get("forces")
                stress = s.properties.get("stress")

                if energy is not None and forces is not None:
                    # Remove labels from info to avoid conflict with Calculator during write
                    atoms.info.pop("energy", None)
                    atoms.info.pop("forces", None)
                    atoms.info.pop("stress", None)

                    calc = SinglePointCalculator(
                        atoms,
                        energy=energy,
                        forces=forces,
                        stress=stress,
                    )
                    atoms.calc = calc

                yield atoms

        # Write to temporary extxyz using the generator
        extxyz_path = output_path.parent / "temp_dataset.extxyz"

        # ASE write supports generators/iterables for many formats including extxyz
        write(extxyz_path, atom_generator(), format="extxyz")  # type: ignore[no-untyped-call, arg-type]

        # Run pace_collect
        cmd = ["pace_collect", str(extxyz_path), str(output_path)]
        logger.info(f"Running conversion: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"pace_collect failed: {e.stderr}")
            raise RuntimeError(f"pace_collect failed: {e.stderr}") from e
        except FileNotFoundError as e:
            # Re-raise with clear message
            raise RuntimeError("pace_collect executable not found.") from e

        return output_path
