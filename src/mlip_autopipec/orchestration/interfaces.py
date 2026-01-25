from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from ase import Atoms

from mlip_autopipec.data_models.dft_models import DFTResult

# Define Protocols for loose coupling (DIP)


class BuilderProtocol(Protocol):
    def build(self) -> Iterator[Atoms]: ...


class SurrogateProtocol(Protocol):
    def run(self) -> None: ...


class DFTRunnerProtocol(Protocol):
    def run(self, atoms: Atoms) -> DFTResult | None: ...


class TrainerProtocol(Protocol):
    def train(
        self, config: Any, dataset_builder: Any, config_gen: Any, work_dir: Path, generation: int
    ) -> Any: ...


class InferenceRunnerProtocol(Protocol):
    def run(self, structures: list[Atoms]) -> Any: ...
