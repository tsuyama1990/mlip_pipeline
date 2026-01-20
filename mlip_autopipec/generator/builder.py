import logging
import uuid

from ase import Atoms
from ase.build import bulk, molecule

from mlip_autopipec.config.schemas.system import SystemConfig
from mlip_autopipec.exceptions import GeneratorException

from .alloy import AlloyGenerator
from .defect import DefectApplicator
from .molecule import MoleculeGenerator

logger = logging.getLogger(__name__)


class StructureBuilder:
    """
    Facade for structure generation. Orchestrates Alloy, Molecule, and Defect generators.
    """

    def __init__(self, system_config: SystemConfig) -> None:
        """
        Initialize the StructureBuilder.

        Args:
            system_config (SystemConfig): The full system configuration.
        """
        self.system_config = system_config
        self.generator_config = system_config.generator_config

        if not self.generator_config:
            logger.info("No generator_config provided in SystemConfig, using defaults.")
            raise GeneratorException("GeneratorConfig is missing in SystemConfig.")

        self.alloy_gen = AlloyGenerator(self.generator_config)
        self.mol_gen = MoleculeGenerator(self.generator_config)
        self.defect_applicator = DefectApplicator(self.generator_config)

    def build(self) -> list[Atoms]:
        """
        Orchestrates the generation process based on TargetSystem.

        The process follows a strict pipeline:
        1. Generate Base Structures (SQS or Molecule).
        2. Apply Distortions (Strain, Rattle, NMS).
        3. Apply Defects (Post-processing).

        Returns:
            List[Atoms]: A list of generated atomic structures.

        Raises:
            GeneratorException: If critical generation steps fail.
        """
        target = self.system_config.target_system
        if not target:
            logger.warning("No target_system defined. Returning empty structure list.")
            return []

        base_structures: list[Atoms] = []

        try:
            # 1. Base Generation Phase
            if target.structure_type == "bulk":
                base_structures = self._generate_bulk_base(target)
            elif target.structure_type == "molecule":
                base_structures = self._generate_molecule_base(target)
            elif target.structure_type == "defect":
                # Treated as bulk with mandatory defects in config
                base_structures = self._generate_bulk_base(target)
            else:
                logger.warning(f"Unknown structure_type: {target.structure_type}")
                return []

            # 2. Defect Application Phase (Post-Processing)
            # Apply defects to the generated base structures via Applicator
            primary_elem = next(iter(target.composition.root.keys()))
            final_structures = self.defect_applicator.apply(base_structures, primary_elem)

        except Exception as e:
            if isinstance(e, GeneratorException):
                raise
            msg = f"Structure generation failed: {e}"
            raise GeneratorException(msg, context={"target": target.name}) from e

        # 3. Final Metadata Tagging
        for s in final_structures:
            if "uuid" not in s.info:
                s.info["uuid"] = str(uuid.uuid4())
            s.info["target_system"] = target.name

        return final_structures

    def _generate_bulk_base(self, target) -> list[Atoms]:
        """Generates bulk structures including SQS and Distortions."""
        structures = []

        # Heuristic: Take the first element from composition and use its bulk structure.
        primary_elem = next(iter(target.composition.root.keys()))
        try:
            prim = bulk(primary_elem)
        except Exception as e:
            logger.warning(
                f"Could not build bulk for {primary_elem}: {e}. Falling back to 'Fe' bcc."
            )
            prim = bulk("Fe")

        # SQS & Distortions
        if self.generator_config.sqs.enabled:
            # Generate SQS
            sqs = self.alloy_gen.generate_sqs(prim, target.composition)

            # Apply Distortions (Strain + Rattle)
            # generate_batch handles the expansion logic
            batch = self.alloy_gen.generate_batch(sqs)
            structures.extend(batch)
        else:
            # If SQS disabled, pass
            pass

        return structures

    def _generate_molecule_base(self, target) -> list[Atoms]:
        """Generates molecular structures including NMS."""
        structures = []
        try:
            mol = molecule(target.name)
        except Exception as e:
            msg = f"Could not build molecule '{target.name}': {e}"
            raise GeneratorException(msg) from e

        if self.generator_config.nms.enabled:
            n_samples = self.generator_config.nms.n_samples
            for T in self.generator_config.nms.temperatures:
                samples = self.mol_gen.normal_mode_sampling(mol, T, n_samples=n_samples)
                structures.extend(samples)
        else:
            structures.append(mol)

        return structures
