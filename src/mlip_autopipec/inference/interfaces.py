"""
This module contains the interfaces for the Inference Engine.
"""

from typing import Protocol

from ase.atoms import Atoms

from mlip_autopipec.config.schemas.inference import InferenceResult


class MDRunner(Protocol):
    """
    Protocol for Molecular Dynamics Runner.
    """

    def run(self, atoms: Atoms) -> InferenceResult:
        """
        Executes a simulation.

        Args:
            atoms: Atomic structure.

        Returns:
            InferenceResult object.
        """
        ...
