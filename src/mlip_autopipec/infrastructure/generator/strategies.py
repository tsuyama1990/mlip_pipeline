import secrets

import numpy as np

from mlip_autopipec.domain_models import Structure
from mlip_autopipec.interfaces import BaseStructureGenerator


class RandomDisplacement(BaseStructureGenerator):
    """
    Generator that perturbs atomic positions by a random vector.
    """
    def generate(self, base_structure: Structure, strategy: str = "random_displacement") -> list[Structure]:
        """
        Generates perturbed structures.
        """
        # Get magnitude from params or default
        magnitude = float(self.params.get("magnitude", 0.01))

        # Create a deep copy
        new_struct = base_structure.model_copy(deep=True)

        # Secure seeding
        seed = secrets.randbits(128)
        rng = np.random.default_rng(seed)

        # Generate random displacements [-magnitude, magnitude]
        displacement = rng.uniform(-magnitude, magnitude, size=new_struct.positions.shape)

        new_struct.positions += displacement

        return [new_struct]
