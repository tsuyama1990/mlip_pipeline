from typing import Iterable, Iterator

from mlip_autopipec.components.oracle.base import BaseOracle
from mlip_autopipec.domain_models.config import QEOracleConfig
from mlip_autopipec.domain_models.structure import Structure


class QEOracle(BaseOracle):
    def __init__(self, config: QEOracleConfig) -> None:
        super().__init__(config)
        self.config: QEOracleConfig = config

    @property
    def name(self) -> str:
        return self.config.name

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        msg = "Quantum Espresso Oracle is not yet implemented."
        raise NotImplementedError(msg)
