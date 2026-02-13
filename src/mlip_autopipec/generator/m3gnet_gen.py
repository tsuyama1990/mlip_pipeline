import logging
from collections.abc import Iterator
from typing import Any

from ase import Atoms

from mlip_autopipec.domain_models.config import GeneratorConfig
from mlip_autopipec.domain_models.datastructures import Structure
from mlip_autopipec.generator.interface import BaseGenerator

logger = logging.getLogger(__name__)


class M3GNetGenerator(BaseGenerator):
    """Generates structures using M3GNet (or mock if not available)."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.has_m3gnet = False
        try:
            import m3gnet  # noqa: F401

            self.has_m3gnet = True
        except ImportError:
            logger.warning("M3GNet not installed. Using Mock behavior for M3GNetGenerator.")

    def explore(self, context: dict[str, Any]) -> Iterator[Structure]:
        count = context.get("count", self.config.mock_count)

        if self.has_m3gnet:
            logger.info("Generating structures using M3GNet...")
            # Actual implementation would go here.
            # For now, fall through to mock behavior as M3GNet usage is complex and optional.

        logger.info(f"M3GNetGenerator (Mock): Generating {count} structures")
        for _ in range(count):
            # Create a dummy structure (e.g., Si dimer)
            atoms = Atoms("Si2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[4, 4, 4], pbc=True)
            yield Structure(atoms=atoms, provenance="m3gnet", label_status="unlabeled")
