import logging
from collections.abc import Iterator

import numpy as np
from ase import Atoms

from mlip_autopipec.config.schemas.generator import DistortionConfig
from mlip_autopipec.generator.transformations import apply_rattle, apply_strain

logger = logging.getLogger(__name__)


class DistortionStrategy:
    """
    Strategy for applying lattice strain and thermal rattling distortions.
    """

    def __init__(self, config: DistortionConfig, seed: int | None = None) -> None:
        """
        Initialize the DistortionStrategy.

        Args:
            config: Distortion configuration.
            seed: Random seed for deterministic generation.
        """
        self.config = config
        self.rng = np.random.default_rng(seed)

    def apply(self, structures: Iterator[Atoms]) -> Iterator[Atoms]:
        """
        Applies distortions to a stream of structures.

        Args:
            structures: Iterator yielding input structures.

        Yields:
            Atoms: Original and distorted structures.
        """
        if not self.config.enabled:
            yield from structures
            return

        n_strain = self.config.n_strain_steps
        n_rattle = self.config.n_rattle_steps
        strain_range = self.config.strain_range
        rattle_stdev = self.config.rattle_stdev

        for base in structures:
            # Yield base structure first
            yield base

            # Strains
            strains = np.linspace(strain_range[0], strain_range[1], n_strain)
            # Store strained structures temporarily to apply rattles on them
            # Since n_strain is usually small (e.g. 5), this is acceptable memory usage per base structure
            strained_pool: list[Atoms] = [base]

            for s in strains:
                if abs(s) < 1e-6:
                    continue  # Skip zero strain (base)

                # Hydrostatic
                strain_tensor = np.eye(3) * s
                try:
                    strained = apply_strain(base, strain_tensor)
                    strained_pool.append(strained)
                    yield strained
                except Exception as e:
                    logger.warning(f"Strain failed: {e}")

            # Rattles
            for st in strained_pool:
                for _ in range(n_rattle):
                    try:
                        rattled = apply_rattle(st, rattle_stdev, rng=self.rng)
                        # Inherit metadata
                        if "strain_tensor" in st.info:
                            rattled.info["strain_tensor"] = st.info["strain_tensor"]
                        rattled.info["parent_config_type"] = st.info.get("config_type")
                        yield rattled
                    except Exception as e:
                        logger.warning(f"Rattle failed: {e}")
