"""DFT Oracle Module."""

from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from ase import Atoms
from ase.calculators.lj import LennardJones
from loguru import logger

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import DFTConfig, PYACEMAKERConfig
from pyacemaker.core.interfaces import Oracle
from pyacemaker.core.utils import update_structure_metadata
from pyacemaker.domain_models.models import StructureMetadata, StructureStatus


class DFTOracle(Oracle):
    """Oracle wrapping DFT calculations (VASP/QE) or Mock."""

    def __init__(self, config: PYACEMAKERConfig | DFTConfig, mock: bool = False) -> None:
        """Initialize the DFT Oracle."""
        # BaseModule expects PYACEMAKERConfig
        self.dft_config: DFTConfig
        self.mock = mock

        if isinstance(config, PYACEMAKERConfig):
            super().__init__(config)
            self.dft_config = config.oracle.dft
            self.mock = config.oracle.mock or mock
        else:
            self.config = config  # type: ignore[assignment]
            self.dft_config = config
            self.logger = logger.bind(name="DFTOracle")
            self.mock = mock

    def run(self) -> ModuleResult:
        """Execute default oracle task (not usually called directly)."""
        return ModuleResult(
            status="success",
            metrics=Metrics(message="DFTOracle ready")  # type: ignore[call-arg]
        )

    def compute_batch(self, structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
        """Compute energy/forces for a batch of structures (Streaming)."""
        workers = self.dft_config.max_workers if not self.mock else 1
        buffer_size = workers * 2

        iterator = iter(structures)
        futures_map = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:

            def submit_next() -> bool:
                try:
                    s = next(iterator)
                    f = executor.submit(self._compute_single, s)
                    futures_map[f] = s
                except StopIteration:
                    return False
                else:
                    return True

            # Fill buffer
            for _ in range(buffer_size):
                if not submit_next():
                    break

            while futures_map:
                done, _ = wait(futures_map.keys(), return_when=FIRST_COMPLETED)

                for f in done:
                    structure = futures_map.pop(f)
                    try:
                        result_atoms = f.result()
                        update_structure_metadata(structure, result_atoms)
                        structure.label_source = "dft"
                    except Exception:
                        self.logger.exception(f"DFT calculation failed for {structure.id}")
                        structure.status = StructureStatus.FAILED

                    yield structure

                    submit_next()

    def _compute_single(self, structure: StructureMetadata) -> Atoms:
        """Compute for a single structure."""
        if "atoms" not in structure.features:
            msg = f"Structure {structure.id} has no atoms."
            raise ValueError(msg)

        atoms = structure.features["atoms"].copy()

        if self.mock:
            # Use Lennard-Jones
            atoms.calc = LennardJones(epsilon=1.0, sigma=2.5)  # type: ignore[no-untyped-call]
            # Trigger calculation
            atoms.get_potential_energy()
            atoms.get_forces()
        else:
            # Real DFT not implemented, fallback to mock with warning
            self.logger.warning("Real DFT not implemented. Using Mock.")
            atoms.calc = LennardJones()  # type: ignore[no-untyped-call]
            atoms.get_potential_energy()

        return atoms  # type: ignore[no-any-return]
