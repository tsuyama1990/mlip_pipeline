from collections.abc import Iterable, Iterator

from mlip_autopipec.components.dynamics.base import BaseDynamics
from mlip_autopipec.domain_models.config import LAMMPSDynamicsConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure


class LAMMPSDynamics(BaseDynamics):
    def __init__(self, config: LAMMPSDynamicsConfig) -> None:
        super().__init__(config)
        self.config: LAMMPSDynamicsConfig = config

    @property
    def name(self) -> str:
        return self.config.name

    def explore(
        self, potential: Potential, start_structures: Iterable[Structure]
    ) -> Iterator[Structure]:
        msg = "LAMMPS Dynamics is not yet implemented."
        raise NotImplementedError(msg)
