import logging
import uuid
from collections.abc import Generator

import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell

from mlip_autopipec.config.models import GeneratorConfig, SystemConfig
from mlip_autopipec.generator.defects import DefectStrategy
from mlip_autopipec.generator.transformations import apply_rattle, apply_strain

logger = logging.getLogger(__name__)


class StructureBuilder:
    """
    Orchestrates the generation of atomic structures based on configuration.
    """

    def __init__(self, config: SystemConfig):
        self.sys_config = config
        self.config: GeneratorConfig = config.generator_config

    def build(self) -> Generator[Atoms, None, None]:
        """
        Generates structures.
        """
        rng = np.random.default_rng(self.config.seed)

        logger.info("Building base structures...")

        generated_count = 0
        limit = self.config.number_of_structures

        # 1. Base Structure
        # If target_system is defined, use it.
        if self.sys_config.target_system:
             primary = self.sys_config.target_system.elements[0]
             structure_type = self.sys_config.target_system.crystal_structure or "fcc"
             try:
                 base_atoms = bulk(primary, structure_type)
             except Exception:
                 base_atoms = bulk(primary, "fcc", a=4.05)
        else:
             # Default fallback
             base_atoms = bulk("Al", "fcc", a=4.05)

        # 2. Supercell
        if self.config.sqs.supercell_size:
            dim = self.config.sqs.supercell_size
            if len(dim) == 3:
                P = np.diag(dim)
                base_atoms = make_supercell(base_atoms, P)

        # Yield base
        base_atoms.info["config_type"] = "base"
        base_atoms.info["uuid"] = str(uuid.uuid4())

        if self._validate(base_atoms):
            yield base_atoms
            generated_count += 1

        if generated_count >= limit:
            return

        # 3. Distortions
        dist_conf = self.config.distortion
        if dist_conf.enabled:
            strains = np.linspace(dist_conf.strain_range[0], dist_conf.strain_range[1], dist_conf.n_strain_steps)

            for eps in strains:
                strain_tensor = np.eye(3) * eps
                try:
                    strained = apply_strain(base_atoms, strain_tensor)
                    strained.info["config_type"] = "strain_vol"
                    strained.info["uuid"] = str(uuid.uuid4())

                    # Only yield pure strain if it is non-zero (avoid duplicating base)
                    if abs(eps) >= 1e-6:
                        if self._validate(strained):
                            yield strained
                            generated_count += 1

                    if generated_count >= limit:
                        return

                    if dist_conf.rattle_stdev > 0:
                        for _ in range(dist_conf.n_rattle_steps):
                            rattled = apply_rattle(strained, dist_conf.rattle_stdev, rng=rng)
                            rattled.info["uuid"] = str(uuid.uuid4())
                            if self._validate(rattled):
                                yield rattled
                                generated_count += 1
                            if generated_count >= limit:
                                return
                except Exception as e:
                    logger.warning(f"Failed to generate strained structure: {e}")

        # 4. Defects
        defect_conf = self.config.defects
        if defect_conf.enabled:
            strategy = DefectStrategy(defect_conf, seed=self.config.seed)
            defect_structures = strategy.apply([base_atoms])
            # The strategy logic applies defects to copies.
            # We skip the first one if it's just the base (depends on impl).
            # Usually apply returns list of modified structures.
            for atoms in defect_structures:
                atoms.info["uuid"] = str(uuid.uuid4())
                if self._validate(atoms):
                    yield atoms
                    generated_count += 1
                if generated_count >= limit:
                    return

    def _validate(self, atoms: Atoms) -> bool:
        if not isinstance(atoms, Atoms):
            logger.error("Generated object is not an ASE Atoms object")
            return False

        if hasattr(atoms, "positions"):
            pos = atoms.get_positions()
            if np.isnan(pos).any() or np.isinf(pos).any():
                logger.error("Structure has NaN/Inf positions")
                return False

        # Check Cell
        if atoms.pbc.any():
            cell = atoms.get_cell()
            if np.linalg.det(cell) == 0:
                 logger.error("Structure has zero cell volume")
                 return False

        return True
