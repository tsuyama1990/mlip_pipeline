import logging
from pathlib import Path

from ase.io import read, write

from mlip_autopipec.config import DFTConfig
from mlip_autopipec.domain_models.structures import CandidateStructure
from mlip_autopipec.physics.oracle.espresso import EspressoRunner

logger = logging.getLogger(__name__)


class DFTManager:
    def __init__(self, config: DFTConfig) -> None:
        self.config = config
        self.runner = EspressoRunner(config)

    def compute(self, candidates: list[CandidateStructure], work_dir: Path) -> list[Path]:
        logger.info(f"Starting DFT calculation for {len(candidates)} candidates.")
        results_paths: list[Path] = []
        for i, candidate in enumerate(candidates):
            try:
                logger.info(f"Computing candidate {i + 1}/{len(candidates)}")
                # Read structure
                atoms_obj = read(candidate.structure_path)
                # Handle list vs single Atoms
                atoms = atoms_obj[0] if isinstance(atoms_obj, list) else atoms_obj

                # run_single returns a new Atoms object with results
                res = self.runner.run_single(atoms)

                # Write result
                output_path = work_dir / f"labeled_{i:04d}.extxyz"
                write(output_path, res)
                results_paths.append(output_path)
            except Exception:
                logger.exception(f"Failed to compute candidate {i + 1}")
                raise

        return results_paths
