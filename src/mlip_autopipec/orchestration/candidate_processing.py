import contextlib
import logging
import tempfile
from pathlib import Path

from ase import Atoms
from ase.io import read, write

from mlip_autopipec.orchestration.database import DatabaseManager
from mlip_autopipec.training.pacemaker import PacemakerWrapper

logger = logging.getLogger(__name__)

class CandidateManager:
    def __init__(self, db_manager: DatabaseManager, pacemaker: PacemakerWrapper) -> None:
        self.db = db_manager
        self.pacemaker = pacemaker

    def process_halted(
        self, dump_path: Path, potential_path: Path, n_perturbations: int = 10, cycle_index: int = 0
    ) -> None:
        """
        Process a halted MD structure: extract, perturb, select, embed, save.
        """
        if not dump_path.exists():
            logger.error(f"Dump file not found: {dump_path}")
            return

        try:
            # 1. Extract
            atoms = self._extract_cluster(dump_path)

            # 2. Perturb
            candidates = self._perturb_structure(atoms, n_perturbations)

            # 3. Select (Active Learning)
            # Write candidates to temp file for pace_activeset
            selected_indices = []
            with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w+", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                try:
                    # type: ignore[no-untyped-call]
                    write(str(tmp_path), candidates, format="extxyz")

                    selected_indices = self.pacemaker.select_active_set(tmp_path, potential_path)
                except Exception:
                    logger.exception("Failed during active set selection")
                finally:
                    if tmp_path.exists():
                        with contextlib.suppress(OSError):
                            tmp_path.unlink()

            logger.info(f"Selected {len(selected_indices)} candidates from {len(candidates)} perturbations.")

            # 4. Embed and Save Selected
            for idx in selected_indices:
                if idx < 0 or idx >= len(candidates):
                    continue

                cand_atoms = candidates[idx]
                embedded = self._embed_structure(cand_atoms)

                # Metadata
                metadata = {
                    "origin_dump": str(dump_path),
                    "perturbation_index": idx,
                    "cycle": cycle_index,
                    "config_type": "active_learning",
                    "status": "pending"  # Ready for DFT
                }

                self.db.add_structure(
                    embedded,
                    metadata=metadata
                )

        except Exception:
            logger.exception(f"Failed to process halted structure: {dump_path}")

    def _extract_cluster(self, dump_path: Path) -> Atoms:
        # For now, read the last frame of the dump
        # type: ignore[no-untyped-call]
        atoms = read(str(dump_path), index=-1)
        if not isinstance(atoms, Atoms):
             # Handle list return if read returns list
             if isinstance(atoms, list):
                 atoms = atoms[-1]
             else:
                 msg = f"Expected Atoms object from {dump_path}"
                 raise TypeError(msg)
        return atoms

    def _perturb_structure(self, atoms: Atoms, n: int, rattle_std: float = 0.1) -> list[Atoms]:
        candidates = []
        for _ in range(n):
            # type: ignore[no-untyped-call]
            new_atoms = atoms.copy()
            # type: ignore[no-untyped-call]
            new_atoms.rattle(stdev=rattle_std)
            candidates.append(new_atoms)
        return candidates

    def _embed_structure(self, atoms: Atoms) -> Atoms:
        # Simple vacuum padding
        # type: ignore[no-untyped-call]
        atoms.center(vacuum=8.0)
        return atoms
