import logging
from collections.abc import Iterable, Iterator

from mlip_autopipec.components.dynamics.base import BaseDynamics
from mlip_autopipec.domain_models.config import LAMMPSDynamicsConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class LAMMPSDynamics(BaseDynamics):
    """
    LAMMPS implementation of the Dynamics component.

    This component runs Molecular Dynamics (MD) simulations using LAMMPS
    to explore the potential energy surface and identify uncertain structures.
    """

    def __init__(self, config: LAMMPSDynamicsConfig) -> None:
        super().__init__(config)
        self.config: LAMMPSDynamicsConfig = config

    @property
    def name(self) -> str:
        return self.config.name

    def explore(
        self, potential: Potential, start_structures: Iterable[Structure]
    ) -> Iterator[Structure]:
        """
        Explore the PES using LAMMPS MD simulations.

        Args:
            potential: The potential to use for MD.
            start_structures: Initial structures for the MD runs.

        Yields:
            Structure: Structures identified as uncertain/extrapolated.

        Raises:
            NotImplementedError: Always, as this is a placeholder.
        """
        logger.error("LAMMPS Dynamics is not yet implemented.")
        msg = "LAMMPS Dynamics is not yet implemented."
        raise NotImplementedError(msg)
