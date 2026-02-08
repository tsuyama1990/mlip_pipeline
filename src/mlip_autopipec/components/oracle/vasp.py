from collections.abc import Iterable, Iterator

from mlip_autopipec.components.oracle.base import BaseOracle
from mlip_autopipec.domain_models.config import VASPOracleConfig
from mlip_autopipec.domain_models.structure import Structure


class VASPOracle(BaseOracle):
    def __init__(self, config: VASPOracleConfig) -> None:
        super().__init__(config)
        self.config: VASPOracleConfig = config

    @property
    def name(self) -> str:
        return self.config.name

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        msg = "VASP Oracle is not yet implemented."
        raise NotImplementedError(msg)
