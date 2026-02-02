from typing import Protocol

from ase import Atoms


class OracleInterface(Protocol):
    def compute(self, structures: list[Atoms]) -> list[Atoms]:
        """
        Takes a list of structures, performs DFT, and returns them
        with 'calc' attached (Energy/Forces/Stress).
        """
        ...
