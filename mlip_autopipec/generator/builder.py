import logging
import uuid
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.exceptions import GeneratorError
from mlip_autopipec.generator.defects import DefectStrategy
from mlip_autopipec.generator.sqs import SQSStrategy
from mlip_autopipec.generator.transformations import apply_rattle, apply_strain

logger = logging.getLogger(__name__)


class StructureBuilder:
    """
    Facade for structure generation strategies.

    This class orchestrates the generation of diverse atomic structures for training ML potentials.
    It integrates three primary generation strategies:
    1.  **SQS (Special Quasirandom Structures)**: Generates disordered supercells for alloys.
    2.  **Distortions**: Applies elastic strain and thermal rattling to explore the potential energy surface.
    3.  **Defects**: Introduces point defects (vacancies, interstitials) to capture defect energetics.

    The builder processes these strategies sequentially to produce a batch of candidate structures.
    """

    def __init__(self, system_config: SystemConfig) -> None:
        """
        Initialize the StructureBuilder with system configuration.

        Args:
            system_config (SystemConfig): The full system configuration object, containing
                                          target system details and generator settings.
        """
        self.system_config = system_config
        self.generator_config = system_config.generator_config

        if not self.generator_config:
            logger.info("No generator_config provided in SystemConfig, using defaults.")
            self.generator_config = GeneratorConfig()

        self.rng = np.random.default_rng(self.generator_config.seed)
        self.sqs_strategy = SQSStrategy(self.generator_config.sqs, seed=self.generator_config.seed)
        self.defect_strategy = DefectStrategy(self.generator_config.defects, seed=self.generator_config.seed)

    def build_batch(self) -> list[Atoms]:
        """
        Orchestrates the generation pipeline to produce a batch of structures.

        The pipeline consists of the following steps:
        1.  **Base Generation**: Creates initial bulk or molecular structures. For alloys,
            this involves generating SQS supercells.
        2.  **Distortion**: Applies lattice strain and atomic rattling to the base structures
            to create a distorted pool.
        3.  **Defect Application**: Introduces vacancies and interstitials to the distorted
            pool.
        4.  **Metadata Tagging**: Assigns unique IDs and system tags to all generated structures.
        5.  **Sampling**: If the number of generated structures exceeds the requested count,
            a random sample is returned.

        Returns:
            list[Atoms]: A list of ASE Atoms objects representing the generated structures.

        Raises:
            GeneratorError: If the generation process fails critically.
        """
        target = self.system_config.target_system
        if not target:
            logger.warning("No target_system defined. Returning empty structure list.")
            return []

        try:
            # 1. Base Generation Phase
            base_structures = self._generate_base(target)

            # 2. Distortions (Strain + Rattle)
            # This returns original + distorted structures
            distorted_structures = self._apply_distortions(base_structures)

            # Use distorted structures (including base) as the pool for defects
            current_pool = distorted_structures

            # 3. Defect Application Phase
            primary_elem = next(iter(target.composition.keys()))
            final_structures = self.defect_strategy.apply(current_pool, primary_elem)

        except Exception as e:
            if isinstance(e, GeneratorError):
                raise
            msg = f"Structure generation failed: {e}"
            logger.error(msg, exc_info=True)
            raise GeneratorError(msg, context={"target": target.name}) from e

        # 4. Final Metadata Tagging
        self._tag_metadata(final_structures, target.name)

        # 5. Limit number of structures if needed (random sample)
        return self._sample_results(final_structures)

    def _tag_metadata(self, structures: list[Atoms], target_name: str) -> None:
        for s in structures:
            if "uuid" not in s.info:
                s.info["uuid"] = str(uuid.uuid4())
            s.info["target_system"] = target_name

    def _sample_results(self, structures: list[Atoms]) -> list[Atoms]:
        n_req = self.generator_config.number_of_structures
        if len(structures) > n_req:
             # Use self.rng to sample indices
             indices = self.rng.choice(len(structures), size=n_req, replace=False)
             return [structures[i] for i in indices]
        return structures

    def _generate_base(self, target: Any) -> list[Atoms]:
        """
        Generates base structures based on target type (bulk/molecule).
        """
        structure_type = "bulk" # Default
        if target.crystal_structure:
            structure_type = "bulk"
        elif hasattr(target, "structure_type") and target.structure_type:
            structure_type = target.structure_type
        elif "molecule" in target.name.lower():
             structure_type = "molecule"

        if structure_type == "bulk":
            return self._generate_bulk_base(target)
        if structure_type == "molecule":
            try:
                mol = molecule(target.name)
                return [mol]
            except Exception:
                 logger.warning(f"Could not build molecule {target.name}")
                 return []

        # Default fallback
        return self._generate_bulk_base(target)

    def _generate_bulk_base(self, target: Any) -> list[Atoms]:
        """
        Generates base bulk structures.

        If SQS is enabled, generates a Special Quasirandom Structure supercell.
        Otherwise, generates a primitive cell.

        Args:
            target: The TargetSystem configuration.

        Returns:
            list[Atoms]: A list containing the base bulk structure(s).
        """
        structures = []

        # Heuristic: Take the first element from composition and use its bulk structure.
        primary_elem = next(iter(target.composition.keys()))
        try:
            prim = bulk(primary_elem, crystalstructure=target.crystal_structure or 'fcc')
        except Exception as e:
            logger.warning(
                f"Could not build bulk for {primary_elem}: {e}. Falling back to 'Fe' bcc."
            )
            prim = bulk("Fe")

        if self.generator_config.sqs.enabled:
            sqs = self.sqs_strategy.generate(prim, target.composition)
            structures.append(sqs)
        else:
            # Just primitive
            structures.append(prim)

        return structures

    def _apply_distortions(self, base_structures: list[Atoms]) -> list[Atoms]:
        """
        Applies strain and rattle distortions to base structures.

        Generates a combinatorial set of structures by applying strain steps
        and then rattling each strained structure.

        Args:
            base_structures (list[Atoms]): The input structures to distort.

        Returns:
            list[Atoms]: A list including the original base structures and the distorted variants.
        """
        results = []

        # Include base structures first
        results.extend(base_structures)

        if not self.generator_config.distortion.enabled:
            return results

        n_strain = self.generator_config.distortion.n_strain_steps
        n_rattle = self.generator_config.distortion.n_rattle_steps
        strain_range = self.generator_config.distortion.strain_range
        rattle_stdev = self.generator_config.distortion.rattle_stdev

        for base in base_structures:
            # Strains
            strains = np.linspace(strain_range[0], strain_range[1], n_strain)
            strained_pool = [base]

            for s in strains:
                if abs(s) < 1e-6:
                    continue # Skip zero strain (base)

                # Hydrostatic
                strain_tensor = np.eye(3) * s
                try:
                    strained = apply_strain(base, strain_tensor)
                    strained_pool.append(strained)
                    results.append(strained)
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
                        results.append(rattled)
                    except Exception as e:
                        logger.warning(f"Rattle failed: {e}")

        return results
