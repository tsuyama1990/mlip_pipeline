import logging
from collections.abc import Iterable, Iterator

from mlip_autopipec.components.oracle.base import BaseOracle
from mlip_autopipec.domain_models.config import VASPOracleConfig
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class VASPOracle(BaseOracle):
    """
    VASP implementation of the Oracle component.

    This component is responsible for running DFT calculations using VASP
    to label structures with energy, forces, and stress.
    """

    def __init__(self, config: VASPOracleConfig) -> None:
        super().__init__(config)
        self.config: VASPOracleConfig = config

    @property
    def name(self) -> str:
        return self.config.name

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """
        Compute labels (energy, forces, stress) for the given structures using VASP.

        Args:
            structures: An iterable of unlabeled Structure objects.

        Yields:
            Structure: Labeled Structure objects.

        Raises:
            NotImplementedError: Always, as this is a placeholder.
        """
        logger.error("VASP Oracle is not yet implemented.")
        msg = "VASP Oracle is not yet implemented."
        raise NotImplementedError(msg)

    def __repr__(self) -> str:
        return f"<VASPOracle(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"VASPOracle({self.name})"
