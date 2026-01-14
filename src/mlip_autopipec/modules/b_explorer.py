
# Copyright (C) 2024-present by the LICENSE file authors.
#
# This file is part of MLIP-AutoPipe.
#
# MLIP-AutoPipe is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MLIP-AutoPipe is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MLIP-AutoPipe.  If not, see <https://www.gnu.org/licenses/>.
"""This module provides the SurrogateExplorer class for structure selection."""
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

from .calculators import get_calculator

if TYPE_CHECKING:
    from mlip_autopipec.schemas.system_config import SystemConfig


def farthest_point_sampling(points: np.ndarray, num_to_select: int) -> list[int]:
    """
    Selects a subset of points using the Farthest Point Sampling (FPS) algorithm.

    Args:
        points: A 2D array of points.
        num_to_select: The number of points to select.

    Returns:
        A list of indices of the selected points.
    """
    selected_indices = []
    if num_to_select > 0:
        # Start with a random point
        first_index = int(np.random.randint(len(points)))
        selected_indices.append(first_index)

        for _ in range(num_to_select - 1):
            distances = np.array(
                [
                    min(np.linalg.norm(points[i] - points[j]) for j in selected_indices)
                    for i in range(len(points))
                ]
            )
            farthest_point_index = int(np.argmax(distances))
            selected_indices.append(farthest_point_index)

    return selected_indices


class SurrogateExplorer:
    """
    A class to explore and select structures using a surrogate model.
    """

    def __init__(self, config: "SystemConfig") -> None:
        self.config = config.surrogate_config
        self.calculator = get_calculator(self.config.calculator)
        self.descriptor = SOAP(
            species=config.user_config.target_system.elements,
            periodic=True,
            r_cut=5.0,
            n_max=8,
            l_max=6,
        )

    def select_structures(self, structures: list[Atoms]) -> list[Atoms]:
        """
        Selects a subset of structures using a surrogate model.

        Args:
            structures: A list of ASE Atoms objects.

        Returns:
            A smaller, more diverse list of ASE Atoms objects.
        """
        # 1. Filtering with MACE
        filtered_structures = []
        for atoms in structures:
            atoms.calc = self.calculator
            energy = atoms.get_potential_energy()  # type: ignore[no-untyped-call]
            if energy < -1.0:  # Simple energy cutoff
                filtered_structures.append(atoms)

        # 2. Farthest Point Sampling
        if not filtered_structures:
            return []

        descriptors = self.descriptor.create(filtered_structures)
        selected_indices = farthest_point_sampling(descriptors, self.config.num_to_select_fps)
        return [filtered_structures[i] for i in selected_indices]
