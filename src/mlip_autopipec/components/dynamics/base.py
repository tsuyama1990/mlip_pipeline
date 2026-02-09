from abc import abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.config import DynamicsConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.interfaces.base_component import BaseComponent


class BaseDynamics(BaseComponent[DynamicsConfig]):
    @property
    def name(self) -> str:
        return "dynamics"

    @abstractmethod
    def explore(
        self,
        potential: Potential,
        start_structures: Iterable[Structure],
        workdir: Path | None = None,
        physics_baseline: dict[str, Any] | None = None,
        cycle: int = 0,
        metrics: dict[str, Any] | None = None,
    ) -> Iterator[Structure]:
        """
        Explore and find uncertain structures.

        Args:
            potential: The potential to use for exploration.
            start_structures: Initial structures to start exploration from.
            workdir: Directory to write exploration files (e.g. MD logs).
            physics_baseline: Optional physics baseline configuration.
            cycle: The current active learning cycle number.
            metrics: Optional metrics from the previous cycle.

        Returns:
            Iterator of uncertain/new structures found.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
