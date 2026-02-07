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
        # Get number of structures to generate
        n_structures = int(self.params.get("n_structures", 1))

        if n_structures < 1:
            msg = "n_structures must be at least 1"
            raise ValueError(msg)

        generated_structures = []

        # Secure seeding
        seed = secrets.randbits(128)
        rng = np.random.default_rng(seed)

        for _ in range(n_structures):
            # Create a deep copy
            new_struct = base_structure.model_copy(deep=True)

            # Generate random displacements [-magnitude, magnitude]
            displacement = rng.uniform(-magnitude, magnitude, size=new_struct.positions.shape)

            new_struct.positions += displacement
            generated_structures.append(new_struct)

        return generated_structures
