import uuid
from typing import List
from ase import Atoms
from ase.build import bulk, molecule

from mlip_autopipec.config.schemas.system import SystemConfig
from .alloy import AlloyGenerator
from .molecule import MoleculeGenerator
from .defect import DefectGenerator

class StructureBuilder:
    def __init__(self, system_config: SystemConfig):
        self.system_config = system_config
        self.generator_config = system_config.generator_config

        if not self.generator_config:
            # Fallback or error?
            # SystemConfig allows None, but builder needs it.
            # Assuming it's populated or we use defaults.
            from mlip_autopipec.config.schemas.generator import GeneratorConfig
            self.generator_config = GeneratorConfig()

        self.alloy_gen = AlloyGenerator(self.generator_config)
        self.mol_gen = MoleculeGenerator(self.generator_config)
        self.defect_gen = DefectGenerator(self.generator_config)

    def build(self) -> List[Atoms]:
        """
        Orchestrates the generation process based on TargetSystem.
        """
        target = self.system_config.target_system
        if not target:
            return []

        structures = []

        if target.structure_type == "bulk":
            # Generate SQS
            # Need primitive cell.
            # If target.name implies something known to ASE (e.g. 'Fe'), use it.
            # Else? MinimalConfig doesn't provide crystal structure details beyond name/composition.
            # We assume 'name' is loadable by ase.build.bulk or we need more info.
            # For "Fe-Ni", we might assume fcc Fe lattice.

            # Heuristic: Take the first element from composition and use its bulk structure.
            # TargetSystem.composition is a RootModel, so we access .root
            primary_elem = list(target.composition.root.keys())[0]
            try:
                prim = bulk(primary_elem)
            except Exception:
                # Fallback
                prim = bulk('Fe') # Default

            # Pass dictionary composition
            sqs = self.alloy_gen.generate_sqs(prim, target.composition.root)

            # Apply Distortions (Strain + Rattle)
            batch = self.alloy_gen.generate_batch(sqs)
            structures.extend(batch)

            # Defects?
            vacancies = self.defect_gen.create_vacancy(sqs)
            structures.extend(vacancies)

        elif target.structure_type == "molecule":
            mol = molecule(target.name)
            for T in self.generator_config.temperatures:
                samples = self.mol_gen.normal_mode_sampling(mol, T)
                structures.extend(samples)

        # Post-processing: Add UUIDs and Metadata
        final_list = []
        for s in structures:
            if 'uuid' not in s.info:
                s.info['uuid'] = str(uuid.uuid4())
            s.info['target_system'] = target.name
            final_list.append(s)

        return final_list
