
from ase import Atoms
from ase.build import bulk
from icet import ClusterSpace

from mlip_autopipec.schemas.system_config import SystemConfig


def _generate_alloy_sqs(
    elements: list[str], composition: Dict[str, float], supercell_size: list[int], crystal_structure: str
) -> list[Atoms]:
    """Generates SQS structures for alloys."""
    # This is a simplified implementation. A real one would need more robust setup.
    prim = bulk(elements[0], crystalstructure=crystal_structure, a=3.6, cubic=True)
    cs = ClusterSpace(prim, [4.0]) # Simplified cluster space
    sqs_generator = SpecialQuasiRandomStructureGenerator(cs, [supercell_size], composition)
    sqs_structures = [sqs.get_atoms() for sqs in sqs_generator.generate(1)]
    return sqs_structures

def generate_structures(config: SystemConfig) -> list[Atoms]:
    """
    Generates a list of ASE Atoms objects based on the system configuration.
    """
    if config.generator_params.generation_type == "alloy_sqs":
        # Placeholder for user_config mapping
        elements = ["Fe", "Ni"]
        composition = {"Fe": 0.5, "Ni": 0.5}
        crystal_structure = "fcc"

        sqs_structures = _generate_alloy_sqs(
            elements, composition, config.generator_params.sqs_supercell_size, crystal_structure
        )

        strained_structures = []
        for atoms in sqs_structures:
            for strain in config.generator_params.strain_magnitudes:
                strained_atoms = atoms.copy()
                strained_atoms.set_cell(atoms.get_cell() * (1 + strain), scale_atoms=True)
                strained_atoms.info["config_type"] = f"sqs_strain_{strain}"
                strained_structures.append(strained_atoms)

        rattled_structures = []
        for atoms in strained_structures:
            rattled_atoms = atoms.copy()
            rattled_atoms.rattle(config.generator_params.rattle_std_dev)
            rattled_atoms.info["config_type"] += f"_rattle_{config.generator_params.rattle_std_dev}"
            rattled_structures.append(rattled_atoms)

        return rattled_structures
    raise NotImplementedError(f"Generation type not supported: {config.generator_params.generation_type}")
