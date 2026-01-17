import logging
import uuid
from typing import List

from ase import Atoms
from ase.build import bulk, molecule

from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.config.schemas.system import SystemConfig
from mlip_autopipec.exceptions import GeneratorError
from .alloy import AlloyGenerator
from .defect import DefectGenerator
from .molecule import MoleculeGenerator

logger = logging.getLogger(__name__)


class StructureBuilder:
    """
    Facade for structure generation. Orchestrates Alloy, Molecule, and Defect generators.
    """

    def __init__(self, system_config: SystemConfig):
        """
        Initialize the StructureBuilder.

        Args:
            system_config (SystemConfig): The full system configuration.
        """
        self.system_config = system_config
        self.generator_config = system_config.generator_config

        if not self.generator_config:
            logger.info("No generator_config provided in SystemConfig, using defaults.")
            self.generator_config = GeneratorConfig()

        self.alloy_gen = AlloyGenerator(self.generator_config)
        self.mol_gen = MoleculeGenerator(self.generator_config)
        self.defect_gen = DefectGenerator(self.generator_config)

    def build(self) -> List[Atoms]:
        """
        Orchestrates the generation process based on TargetSystem.

        Returns:
            List[Atoms]: A list of generated atomic structures.

        Raises:
            GeneratorError: If critical generation steps fail.
        """
        target = self.system_config.target_system
        if not target:
            logger.warning("No target_system defined. Returning empty structure list.")
            return []

        structures: List[Atoms] = []

        try:
            if target.structure_type == "bulk":
                self._build_bulk(structures, target)
            elif target.structure_type == "molecule":
                self._build_molecule(structures, target)
            elif target.structure_type == "defect":
                # Explicit defect generation mode if needed,
                # though usually defects are applied to bulk.
                # For now, treat as bulk + defects.
                self._build_bulk(structures, target)
            else:
                logger.warning(f"Unknown structure_type: {target.structure_type}")

        except Exception as e:
            if isinstance(e, GeneratorError):
                raise
            raise GeneratorError(f"Structure generation failed: {e}") from e

        # Post-processing: Add UUIDs and Metadata
        final_list = []
        for s in structures:
            if 'uuid' not in s.info:
                s.info['uuid'] = str(uuid.uuid4())
            s.info['target_system'] = target.name
            final_list.append(s)

        return final_list

    def _build_bulk(self, structures: List[Atoms], target) -> None:
        """Helper to build bulk structures."""
        # Heuristic: Take the first element from composition and use its bulk structure.
        # This assumes standard crystal structures. In a real scenario, user might provide CIF/POSCAR.
        primary_elem = list(target.composition.root.keys())[0]
        try:
            prim = bulk(primary_elem)
        except Exception as e:
            logger.warning(f"Could not build bulk for {primary_elem}: {e}. Falling back to 'Fe' bcc.")
            prim = bulk('Fe')

        # 1. SQS Generation
        # Only if enabled in config (checked inside generator, but also check here to avoid unnecessary calls)
        if self.generator_config.sqs.enabled:
            sqs = self.alloy_gen.generate_sqs(prim, target.composition.root)

            # 2. Distortions (Strain + Rattle)
            # generate_batch checks config.distortion.enabled
            batch = self.alloy_gen.generate_batch(sqs)
            structures.extend(batch)

            # 3. Defects
            # Defects are typically applied to the relaxed base structure (SQS)
            if self.generator_config.defects.enabled:
                if self.generator_config.defects.vacancies:
                    vacancies = self.defect_gen.create_vacancy(sqs)
                    structures.extend(vacancies)

                if self.generator_config.defects.interstitials:
                    # Determine elements to insert. Config might specify list.
                    elements_to_insert = self.generator_config.defects.interstitial_elements
                    if not elements_to_insert:
                        # Default to primary element if not specified
                        elements_to_insert = [primary_elem]

                    for el in elements_to_insert:
                        interstitials = self.defect_gen.create_interstitial(sqs, el)
                        structures.extend(interstitials)

    def _build_molecule(self, structures: List[Atoms], target) -> None:
        """Helper to build molecular structures."""
        try:
            mol = molecule(target.name)
        except Exception as e:
             raise GeneratorError(f"Could not build molecule '{target.name}': {e}") from e

        if self.generator_config.nms.enabled:
            n_samples = self.generator_config.nms.n_samples
            for T in self.generator_config.nms.temperatures:
                samples = self.mol_gen.normal_mode_sampling(mol, T, n_samples=n_samples)
                structures.extend(samples)
        else:
            # If NMS disabled, just return the equilibrium molecule
            structures.append(mol)
