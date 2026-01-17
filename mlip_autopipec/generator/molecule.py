import logging
from typing import List

from ase import Atoms
from mlip_autopipec.config.schemas.generator import GeneratorConfig

logger = logging.getLogger(__name__)

class MoleculeGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config

    def normal_mode_sampling(self, atoms: Atoms, temperature: int, n_samples: int = 5) -> List[Atoms]:
        """
        Generates distorted structures using Normal Mode Sampling.
        """
        import shutil
        import tempfile
        from pathlib import Path

        import numpy as np
        from ase.calculators.emt import EMT
        from ase.vibrations import Vibrations

        # Check if calculator is present, else assign EMT
        if atoms.calc is None:
            try:
                atoms.calc = EMT()
            except Exception:
                logger.warning("No calculator and EMT not available. Cannot perform NMS.")
                return []

        # Run Vibrations
        # We need a temporary directory for displacement files
        work_dir = Path(tempfile.mkdtemp(prefix="nms_"))

        try:
            # We must change directory or set name to path because Vibrations writes to files
            vib = Vibrations(atoms, name=str(work_dir / 'vib'), delta=0.01)
            vib.run()

            # Get vibrations
            vib_energies = vib.get_energies()
            # vib.get_modes() doesn't exist? Use get_mode(i)
            # Or vib.modes attribute?
            # ASE documentation says: get_mode(n) returns mode n.

            # NMS logic
            results = []
            kb = 8.617e-5 # eV/K

            # Filter imaginary modes (negative energies squared)
            # vib.get_energies() returns real energies for stable modes, imaginary (complex) for unstable.

            real_indices = [i for i, e in enumerate(vib_energies) if isinstance(e, float) and e > 1e-3]

            if not real_indices:
                logger.warning("No stable vibrational modes found.")
                return []

            # Pre-fetch modes to avoid reading files in loop if possible,
            # or just call get_mode(i) inside loop.
            # get_mode(i) reads from pickle files usually.

            loaded_modes = {i: vib.get_mode(i) for i in real_indices}

            for _ in range(n_samples):
                disp = np.zeros_like(atoms.positions)
                for mode_idx in real_indices:
                    # Heuristic: Random displacement along mode scaled by T.
                    # Typical bond vibration ~ 0.1 Angstrom at room temp.
                    scale = 0.05 * (temperature / 300)**0.5

                    # Random coefficient
                    c = np.random.normal(0, scale)
                    disp += c * loaded_modes[mode_idx]

                new_atoms = atoms.copy()
                new_atoms.positions += disp
                new_atoms.info['config_type'] = 'nms'
                new_atoms.info['temperature'] = temperature
                results.append(new_atoms)

        except Exception as e:
            logger.error(f"NMS failed: {e}")
            return []
        finally:
            # Cleanup
            if work_dir.exists():
                shutil.rmtree(work_dir)

        return results
