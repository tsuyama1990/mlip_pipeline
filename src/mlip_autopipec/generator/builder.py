import logging
import uuid
from typing import Any, Iterator

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.exceptions import GeneratorError
from mlip_autopipec.generator.defects import DefectStrategy
from mlip_autopipec.generator.sqs import SQSStrategy
from mlip_autopipec.generator.distortions import DistortionStrategy

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
        self.distortion_strategy = DistortionStrategy(self.generator_config.distortion, seed=self.generator_config.seed)

    def build(self) -> Iterator[Atoms]:
        """
        Orchestrates the generation pipeline to produce a stream of structures.
        Implements BuilderProtocol.

        The pipeline consists of the following steps:
        1.  **Base Generation**: Creates initial bulk or molecular structures. For alloys,
            this involves generating SQS supercells.
        2.  **Distortion**: Applies lattice strain and atomic rattling to the base structures
            to create a distorted pool.
        3.  **Defect Application**: Introduces vacancies and interstitials to the distorted
            pool.
        4.  **Metadata Tagging**: Assigns unique IDs and system tags to all generated structures.
        5.  **Sampling**: If the number of generated structures exceeds the requested count,
            reservoir sampling is used to limit the output.

        Yields:
            Atoms: ASE Atoms objects representing the generated structures.

        Raises:
            GeneratorError: If the generation process fails critically.
        """
        target = self.system_config.target_system
        if not target:
            logger.warning("No target_system defined. Returning empty structure stream.")
            return

        try:
            # 1. Base Generation Phase
            base_structures = self._generate_base(target)

            # 2. Distortions (Strain + Rattle)
            # This yields original + distorted structures lazily
            # We wrap base_structures (list) into an iterator
            distorted_stream = self.distortion_strategy.apply(iter(base_structures))

            # 3. Defect Application Phase
            # DefectStrategy.apply currently expects list, we need to adapt it or iterate
            # Since defects might multiply structures, it's better if DefectStrategy supports streaming too.
            # But for now, let's iterate and call apply on single items or refactor DefectStrategy.
            # Refactoring DefectStrategy is cleaner. For now, let's wrap:

            primary_elem = next(iter(target.composition.keys()))

            def defect_generator(stream: Iterator[Atoms]) -> Iterator[Atoms]:
                for s in stream:
                    # apply returns list[Atoms] (original + defects)
                    yield from self.defect_strategy.apply([s], primary_elem)

            defect_stream = defect_generator(distorted_stream)

            # 4. Final Metadata Tagging & 5. Sampling
            yield from self._sample_results(self._tag_metadata_stream(defect_stream, target.name))

        except Exception as e:
            if isinstance(e, GeneratorError):
                raise
            msg = f"Structure generation failed: {e}"
            logger.error(msg, exc_info=True)
            raise GeneratorError(msg, context={"target": target.name}) from e

    def _tag_metadata_stream(self, structures: Iterator[Atoms], target_name: str) -> Iterator[Atoms]:
        for s in structures:
            if "uuid" not in s.info:
                s.info["uuid"] = str(uuid.uuid4())
            s.info["target_system"] = target_name
            yield s

    def _sample_results(self, structures: Iterator[Atoms]) -> Iterator[Atoms]:
        """
        Applies reservoir sampling to limit the number of yielded structures
        without loading everything into memory.
        """
        n_req = self.generator_config.number_of_structures

        # Reservoir sampling
        reservoir: list[Atoms] = []
        for i, s in enumerate(structures):
            if i < n_req:
                reservoir.append(s)
            else:
                j = self.rng.integers(0, i + 1)
                if j < n_req:
                    reservoir[j] = s

        yield from reservoir

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
