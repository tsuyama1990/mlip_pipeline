import numpy as np

from mlip_autopipec.domain_models.structure import Structure


class RattleTransform:
    def __init__(self, stdev: float = 0.01) -> None:
        self.stdev = stdev

    def apply(self, structure: Structure) -> Structure:
        """Apply random rattle to atomic positions."""
        s = structure.model_deep_copy()
        noise = np.random.normal(0, self.stdev, s.positions.shape)
        s.positions += noise
        return s

    def __repr__(self) -> str:
        return f"<RattleTransform(stdev={self.stdev})>"

    def __str__(self) -> str:
        return f"RattleTransform(stdev={self.stdev})"


class StrainTransform:
    def __init__(self, strain_range: float = 0.05) -> None:
        self.strain_range = strain_range

    def apply(self, structure: Structure) -> Structure:
        """Apply random strain to the unit cell."""
        s = structure.model_deep_copy()

        # Create random strain tensor (symmetric)
        # E = 0.5 * (F^T F - I) -> roughly F = I + E for small strain
        # We just perturb cell vectors: cell_new = cell_old * (I + strain)

        strain = (np.random.rand(3, 3) - 0.5) * 2 * self.strain_range
        # Symmetrize for pure strain (optional, but good for physical realism)
        strain = (strain + strain.T) / 2

        deformation = np.eye(3) + strain
        s.cell = np.dot(s.cell, deformation)

        # Also move atoms if fractional coords stay same?
        # ASE/Structure stores Cartesian. So we must update positions too.
        # But for 'Strain', we usually imply affine transformation.
        # s.positions = s.positions * deformation?
        # Yes, strictly speaking.
        s.positions = np.dot(s.positions, deformation)

        return s

    def __repr__(self) -> str:
        return f"<StrainTransform(range={self.strain_range})>"

    def __str__(self) -> str:
        return f"StrainTransform(range={self.strain_range})"
