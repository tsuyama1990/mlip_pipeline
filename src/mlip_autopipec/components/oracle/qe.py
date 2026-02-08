import logging
from collections.abc import Iterable, Iterator

from mlip_autopipec.components.oracle.base import BaseOracle
from mlip_autopipec.domain_models.config import QEOracleConfig
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class QEOracle(BaseOracle):
    """
    Quantum Espresso (QE) implementation of the Oracle component.

    This component is responsible for running DFT calculations using Quantum Espresso
    to label structures with energy, forces, and stress.
    """

    def __init__(self, config: QEOracleConfig) -> None:
        super().__init__(config)
        self.config: QEOracleConfig = config

    @property
    def name(self) -> str:
        return self.config.name

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """
        Compute labels (energy, forces, stress) for the given structures using QE.

        Args:
            structures: An iterable of unlabeled Structure objects.

        Yields:
            Structure: Labeled Structure objects.

        Raises:
            NotImplementedError: Always, as this is a placeholder.
        """
        logger.error("Quantum Espresso Oracle is not yet implemented.")
        msg = "Quantum Espresso Oracle is not yet implemented."
        raise NotImplementedError(msg)
