"""Exploration strategies for structure generation."""

from abc import ABC, abstractmethod
from typing import Any, cast

from ase import Atoms
from loguru import logger

from pyacemaker.generator.mutations import apply_strain, create_vacancy, rattle_atoms


class ExplorationStrategy(ABC):
    """Abstract base class for exploration strategies."""

    @abstractmethod
    def generate(self, seed: Atoms, n_candidates: int, **kwargs: Any) -> list[Atoms]:
        """Generate candidate structures from a seed.

        Args:
            seed: Initial structure.
            n_candidates: Number of candidates to generate.
            **kwargs: Additional parameters.

        Returns:
            List of generated structures.
        """


class RandomStrategy(ExplorationStrategy):
    """Generates candidates by applying random strain and rattling."""

    def __init__(self, strain_range: float = 0.1, rattle_amplitude: float = 0.1) -> None:
        """Initialize with perturbation parameters."""
        self.strain_range = strain_range
        self.rattle_amplitude = rattle_amplitude

    def generate(self, seed: Atoms, n_candidates: int, **kwargs: Any) -> list[Atoms]:
        """Generate candidates using random perturbation."""
        candidates = []
        for _ in range(n_candidates):
            new_atoms = apply_strain(seed, self.strain_range)
            new_atoms = rattle_atoms(new_atoms, self.rattle_amplitude)
            candidates.append(new_atoms)
        return candidates


class DefectStrategy(ExplorationStrategy):
    """Generates candidates by introducing defects into supercells."""

    def __init__(self, defect_density: float = 0.01) -> None:
        """Initialize with target defect density."""
        self.defect_density = defect_density

    def generate(self, seed: Atoms, n_candidates: int, **kwargs: Any) -> list[Atoms]:
        """Generate candidates with defects."""
        candidates = []
        for _ in range(n_candidates):
            # Calculate number of defects based on density
            n_atoms = len(seed)
            n_defects = max(1, round(n_atoms * self.defect_density))

            new_atoms = seed.copy()  # type: ignore[no-untyped-call]
            # Simple vacancy loop
            for _ in range(n_defects):
                new_atoms = create_vacancy(cast(Atoms, new_atoms))
                if len(new_atoms) == 0:
                    break

            candidates.append(cast(Atoms, new_atoms))
        return candidates


class M3GNetStrategy(ExplorationStrategy):
    """Uses M3GNet universal potential for relaxation (mockable)."""

    def __init__(self, fallback_strategy: ExplorationStrategy | None = None) -> None:
        """Initialize with fallback strategy."""
        self.fallback_strategy = fallback_strategy or RandomStrategy(
            strain_range=0.05, rattle_amplitude=0.05
        )

    def generate(self, seed: Atoms, n_candidates: int, **kwargs: Any) -> list[Atoms]:
        """Generate candidates by relaxing with M3GNet."""
        try:
            from m3gnet.models import Relaxer
        except ImportError:
            logger.warning("M3GNet not installed. Using fallback strategy.")
            return self.fallback_strategy.generate(seed, n_candidates, **kwargs)

        try:
            relaxer = Relaxer()  # potential=DEFAULT_POTENTIAL
        except Exception as e:
            logger.warning(f"Failed to initialize M3GNet Relaxer: {e}")
            return self.fallback_strategy.generate(seed, n_candidates, **kwargs)

        candidates = []

        # Perturb slightly then relax to explore basins
        perturb_strategy = RandomStrategy(strain_range=0.1, rattle_amplitude=0.1)
        perturbed_candidates = perturb_strategy.generate(seed, n_candidates)

        for p in perturbed_candidates:
            try:
                relaxed = relaxer.relax(p)
                # M3GNet returns a dict with 'final_structure' as Atoms object or similar
                candidates.append(relaxed["final_structure"])
            except Exception as e:
                logger.warning(f"M3GNet relaxation failed: {e}")
                candidates.append(p)  # Fallback to unrelaxed

        return candidates
