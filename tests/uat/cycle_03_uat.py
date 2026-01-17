import numpy as np
import pytest
from ase.build import bulk, molecule
from pathlib import Path
import logging
from unittest.mock import MagicMock, patch

from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.generator.alloy import AlloyGenerator
from mlip_autopipec.generator.molecule import MoleculeGenerator
from mlip_autopipec.generator.defect import DefectGenerator
from mlip_autopipec.generator.builder import StructureBuilder
from mlip_autopipec.config.schemas.system import SystemConfig, TargetSystem
from mlip_autopipec.config.schemas.common import MinimalConfig

# UAT Scenarios implementation
# These tests verify the "User Experience" flow defined in UAT.md

def setup_uat_logging():
    logging.basicConfig(level=logging.INFO)

@pytest.fixture
def uat_alloy_generator():
    config = GeneratorConfig()
    config.sqs.supercell_matrix = [[2,0,0], [0,2,0], [0,0,2]] # 8 atoms
    config.distortion.rattling_amplitude = 0.1
    config.distortion.strain_range = (-0.1, 0.1)
    config.distortion.n_strain_steps = 3 # -0.1, 0.0, 0.1

    return AlloyGenerator(config)

def test_uat_03_01_alloy_sqs_generation(uat_alloy_generator):
    """
    UAT-03-01: Alloy SQS Generation
    Verify that the system can generate a Special Quasirandom Structure (SQS)
    for a specified binary composition (e.g., Fe-Ni 50:50) that matches the target stoichiometry exactly.
    """
    prim = bulk('Fe', 'fcc', a=3.5)
    composition = {"Fe": 0.5, "Ni": 0.5}

    sqs = uat_alloy_generator.generate_sqs(prim, composition)

    # 8 atoms total (2x2x2 supercell)
    assert len(sqs) == 8

    # Stoichiometry check
    symbols = sqs.get_chemical_symbols()
    assert symbols.count('Fe') == 4
    assert symbols.count('Ni') == 4

    # Metadata check
    assert sqs.info['config_type'] == 'sqs'
    assert sqs.pbc.all()

def test_uat_03_02_distortion_pipeline(uat_alloy_generator):
    """
    UAT-03-02: Distortion Pipeline
    Verify that the system can take a base structure and generate a "family" of distorted structures
    including volumetric strain and thermal rattling.
    """
    prim = bulk('Fe', 'fcc', a=3.5)
    composition = {"Fe": 0.5, "Ni": 0.5}
    sqs = uat_alloy_generator.generate_sqs(prim, composition)

    # Generate batch
    batch = uat_alloy_generator.generate_batch(sqs)

    # Expected count:
    # 1 Base
    # Strains: -0.1, 0.0, 0.1. 0.0 skipped (dup of base). So 2 strains.
    # Rattles: 3 base structures (Base + 2 Strains).
    # n_rattle_steps default is 3.
    # Total = 3 + (3 * 3) = 12 structures.

    assert len(batch) == 1 + 2 + 9

    types = [s.info.get('config_type') for s in batch]
    assert 'sqs' in types # Or 'base' if reassigned, but SQS has type 'sqs' initially
    assert 'strain' in types
    assert 'rattle' in types

    # Verify Volume Change for strain
    # Strain 0.1 means Volume ~ 1.33x
    strained_structs = [s for s in batch if s.info.get('config_type') == 'strain']
    base_vol = sqs.get_volume()
    # Check max volume
    max_vol = max(s.get_volume() for s in strained_structs)
    assert max_vol > base_vol

def test_uat_03_03_molecule_nms():
    """
    UAT-03-03: Molecule NMS
    Verify that for a molecular system (e.g., H2O), the system generates distorted geometries.
    """
    config = GeneratorConfig()
    mol_gen = MoleculeGenerator(config)
    h2o = molecule('H2O')

    # Patch ase.vibrations.Vibrations
    with patch('ase.vibrations.Vibrations') as MockVib:
        instance = MockVib.return_value
        instance.get_energies.return_value = [0.0]*6 + [0.1, 0.2, 0.3]
        modes = [np.zeros((3,3)) for _ in range(9)]
        instance.get_mode.side_effect = lambda i: modes[i]

        h2o.calc = MagicMock()

        samples = mol_gen.normal_mode_sampling(h2o, 300, n_samples=2)
        assert len(samples) == 2
        assert samples[0].info['config_type'] == 'nms'

def test_uat_03_04_metadata_integrity():
    """
    UAT-03-04: Metadata Integrity
    Verify that every generated structure includes the correct metadata.
    """
    gen_config = GeneratorConfig()
    target = TargetSystem(name="Fe", composition={"Fe": 1.0}, elements=["Fe"], structure_type="bulk")
    minimal = MinimalConfig(project_name="uat", target_system=target, resources={"dft_code": "quantum_espresso", "parallel_cores": 1})

    sys_config = MagicMock(spec=SystemConfig)
    sys_config.generator_config = gen_config
    sys_config.target_system = target
    sys_config.minimal = minimal

    builder = StructureBuilder(sys_config)

    structures = builder.build()

    assert len(structures) > 0
    for s in structures:
        assert 'uuid' in s.info
        assert 'target_system' in s.info
        assert s.info['target_system'] == "Fe"
        assert 'config_type' in s.info

if __name__ == "__main__":
    setup_uat_logging()
    pass
