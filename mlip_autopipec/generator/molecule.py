import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms

from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.exceptions import GeneratorError

logger = logging.getLogger(__name__)


class MoleculeGenerator:
    """
    Generator for molecular structures using Normal Mode Sampling (NMS).
    """

    def __init__(self, config: GeneratorConfig) -> None:
        """
        Initialize the MoleculeGenerator.

        Args:
            config (GeneratorConfig): The generator configuration.
        """
        self.config = config

    def normal_mode_sampling(
        self, atoms: Atoms, temperature: int, n_samples: int = 5
    ) -> list[Atoms]:
        """
        Generates distorted structures using Normal Mode Sampling.

        Calculates vibrational modes using a calculator (defaults to EMT if none provided)
        and samples random displacements along these modes corresponding to the given temperature.

        Args:
            atoms (Atoms): The base molecular structure.
            temperature (int): Temperature in Kelvin.
            n_samples (int): Number of distorted structures to generate.

        Returns:
            List[Atoms]: A list of distorted structures.

        Raises:
            GeneratorError: If NMS calculation fails.
        """
        if not self.config.nms.enabled:
            return []

        from ase.calculators.emt import EMT
        from ase.vibrations import Vibrations

        # Check if calculator is present, else assign EMT
        if atoms.calc is None:
            try:
                atoms.calc = EMT()
            except Exception as e:
                logger.warning("No calculator attached and EMT not available. Cannot perform NMS.")
                raise GeneratorError("NMS requires a calculator (or EMT availability).") from e

        # Run Vibrations
        # We need a temporary directory for displacement files
        work_dir = Path(tempfile.mkdtemp(prefix="nms_"))

        try:
            # We must change directory or set name to path because Vibrations writes to files
            # in the current directory or 'name' path.
            vib_name = str(work_dir / "vib")
            vib = Vibrations(atoms, name=vib_name, delta=0.01)
            vib.run()

            # Get vibrations
            vib_energies = vib.get_energies()

            # Filter imaginary modes (negative energies squared)
            # vib.get_energies() returns real energies for stable modes, imaginary (complex) for unstable.
            real_indices = [
                i for i, e in enumerate(vib_energies) if isinstance(e, float) and e > 1e-3
            ]

            if not real_indices:
                logger.warning("No stable vibrational modes found.")
                return []

            # Pre-fetch modes to avoid reading files in loop
            loaded_modes = {i: vib.get_mode(i) for i in real_indices}

            results = []
            for _ in range(n_samples):
                disp = np.zeros_like(atoms.positions)
                for mode_idx in real_indices:
                    # Heuristic: Random displacement along mode scaled by T.
                    # Typical bond vibration ~ 0.1 Angstrom at room temp.
                    # We use a simplified model as per Spec/heuristic.
                    scale = 0.05 * (temperature / 300) ** 0.5

                    # Random coefficient
                    c = np.random.normal(0, scale)
                    disp += c * loaded_modes[mode_idx]

                new_atoms = atoms.copy()
                new_atoms.positions += disp
                new_atoms.info["config_type"] = "nms"
                new_atoms.info["temperature"] = temperature
                results.append(new_atoms)

            return results

        except Exception as e:
            if isinstance(e, GeneratorError):
                raise
            msg = f"NMS calculation failed: {e}"
            raise GeneratorError(msg) from e
        finally:
            # Cleanup
            if work_dir.exists():
                # Ignore errors during cleanup to avoid masking original error
                shutil.rmtree(work_dir, ignore_errors=True)
