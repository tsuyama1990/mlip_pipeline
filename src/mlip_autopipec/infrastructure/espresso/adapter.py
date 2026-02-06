import logging
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import CalculatorError
from ase.calculators.espresso import Espresso
from ase.io import iread, write

from mlip_autopipec.config import OracleConfig
from mlip_autopipec.domain_models import Dataset
from mlip_autopipec.infrastructure.espresso.recovery import RecoveryStrategy
from mlip_autopipec.interfaces import BaseOracle

logger = logging.getLogger(__name__)


class EspressoOracle(BaseOracle):
    def __init__(self, config: OracleConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir
        self._validate_command(self.config.command)

    def _validate_command(self, command: str | None) -> None:
        if not command:
            return
        # Allow only alphanumeric, space, dot, dash, underscore, slash
        # Reject anything else to prevent shell injection (e.g. ;, &, |, >, <)
        if not re.match(r"^[\w\s\-\./]+$", command):
            msg = f"Security check failed: Command '{command}' contains illegal characters."
            raise ValueError(msg)

    def label(self, dataset: Dataset) -> Dataset:
        # Create output file
        output_file = self.work_dir / f"labeled_{uuid.uuid4().hex}.extxyz"
        logger.info(f"EspressoOracle: Labeling structures from {dataset.file_path} to {output_file}")

        # Ensure input file exists and is not empty
        if not dataset.file_path.exists() or dataset.file_path.stat().st_size == 0:
            logger.warning(
                f"Input file {dataset.file_path} is missing or empty. Returning empty dataset."
            )
            # Touch the output file so it exists
            output_file.touch()
            return Dataset(file_path=output_file)

        # Use a single temporary directory for the batch to manage files
        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmp_dir:
            tmp_path = Path(tmp_dir)
            count = 0
            success_count = 0

            # Iterate over structures lazily
            # index=None implies iterating over all images for formats that support it?
            # Or iread(..., index=':')?
            # Memory says: "strict lazy loading requires omitting the index".
            # So we use iread(path) which defaults to lazy iteration for supported formats.
            # We assume extxyz or xyz support lazy reading.
            try:
                # Type ignore because iread is not fully typed in ASE stubs or lack thereof
                iterator = iread(dataset.file_path)

                for atoms in iterator:
                    if not isinstance(atoms, Atoms):
                        continue

                    count += 1

                    # Recovery Strategy
                    recovery = RecoveryStrategy()
                    current_params: dict[str, Any] = {}
                    retries = 0
                    success = False

                    while True:
                        try:
                            # Configure Calculator
                            input_data = self.config.scf_params.copy()
                            input_data.update(current_params)

                            # Ensure forces and stress are calculated
                            input_data["tprnfor"] = True
                            input_data["tstress"] = True

                            # Make a unique directory for this calculation inside tmp_dir
                            # to avoid overlapping files if ASE doesn't clean up perfectly
                            # or just use tmp_path. ASE Espresso writes to 'espresso.pwi' etc.
                            # Reusing the same directory is fine if sequential.

                            calc = Espresso(
                                command=self.config.command,
                                pseudopotentials=self.config.pseudopotentials,
                                pseudo_dir=str(self.config.pseudo_dir),
                                kspacing=self.config.kspacing,
                                input_data=input_data,
                                directory=str(tmp_path),
                                tprnfor=True,  # redundant but safe
                                tstress=True,  # redundant but safe
                            )  # type: ignore[no-untyped-call]

                            atoms.calc = calc

                            # Trigger calculation
                            atoms.get_potential_energy()  # type: ignore[no-untyped-call]

                            # Explicitly retrieve results to ensure they are available
                            # and attached to the atoms object for writing
                            # (ASE should handle this automatically when writing atoms with calc)

                            success = True
                            break

                        except CalculatorError as e:
                            logger.warning(f"SCF failed for structure {count} (Attempt {retries}): {e}")

                            recipe = recovery.get_recipe(retries)
                            if recipe is None:
                                logger.exception(
                                    f"Structure {count} failed all recovery attempts. Skipping."
                                )
                                break

                            logger.info(f"Retrying structure {count} with recovery recipe: {recipe}")
                            current_params.update(recipe)
                            retries += 1
                        except Exception:
                            # Catch-all for other errors (IO, etc)
                            logger.exception(f"Unexpected error processing structure {count}")
                            break

                    if success:
                        # Append to output file
                        # Note: append=True ensures we don't overwrite previous structures
                        write(output_file, atoms, append=True)
                        success_count += 1

            except Exception as e:
                logger.exception(f"Failed to read or process input file {dataset.file_path}")
                # We do not raise here, we return what we have so far?
                # Or raise? If input is bad, we probably should raise.
                msg = f"Oracle failed processing {dataset.file_path}"
                raise RuntimeError(msg) from e

        logger.info(f"EspressoOracle: Processed {count} structures, labeled {success_count}.")

        return Dataset(file_path=output_file)
