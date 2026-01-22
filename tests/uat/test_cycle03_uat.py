from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline


@pytest.fixture
def uat_db(tmp_path):
    db_path = tmp_path / "uat.db"
    db = DatabaseManager(db_path)
    db.initialize()
    return db

def test_scenario_3_1_prescreening(uat_db):
    """
    Scenario 3.1: Pre-screening with Foundation Model (using Mock for stability)
    """
    # Pre-conditions: DB contains 50 pending structures
    with uat_db:
        for i in range(50):
            # Create dummy atoms
            atoms = Atoms('Cu2', positions=[[0,0,0], [0,0,2.5]])
            uat_db.save_candidate(atoms, metadata={"status": "pending", "generation": 0})

        assert uat_db.count(selection="status=pending") == 50

        # Steps: Run selection
        config = SurrogateConfig(
            model_type="mock",
            n_samples=10,
            force_threshold=100.0
        )

        pipeline = SurrogatePipeline(uat_db, config)
        pipeline.run()

        # Post-conditions
        assert uat_db.count(selection="status=selected") == 10
        # The rest should be held (40) or rejected (0 if mock force is low)
        # Mock model returns random forces -0.1 to 0.1, so no rejection.
        assert uat_db.count(selection="status=held") == 40

        # Verify metadata
        # We need to fetch one selected atom and check info
        # get_atoms returns Atoms with info populated from key_value_pairs and data?
        # ase.db Row.toatoms() populates info with key_value_pairs.
        selected = uat_db.get_atoms(selection="status=selected")
        # Check if mace_energy is in info
        assert "mace_energy" in selected[0].info

def test_scenario_3_2_filtering(uat_db):
    """
    Scenario 3.2: Filtering Bad Structures (Clash Detection)
    """
    with uat_db:
        # Pre-conditions: User manually inserts a structure with two atoms at distance 0.1 A

        mock_model = MagicMock()

        # 1 valid, 1 invalid
        atoms_valid = Atoms('Cu2', positions=[[0,0,0], [0,0,2.5]])
        atoms_invalid = Atoms('Cu2', positions=[[0,0,0], [0,0,0.1]]) # Clash

        uat_db.save_candidate(atoms_valid, metadata={"status": "pending"})
        uat_db.save_candidate(atoms_invalid, metadata={"status": "pending"})

        # Mock behavior: return low force for valid, high for invalid
        def side_effect(atoms_list):
            energies = np.zeros(len(atoms_list))
            forces = []
            for at in atoms_list:
                dist = at.get_distance(0, 1)
                if dist < 0.5:
                    forces.append(np.ones((2,3)) * 1000.0) # > 50
                else:
                    forces.append(np.zeros((2,3)))
            return energies, forces

        mock_model.compute_energy_forces.side_effect = side_effect
        mock_model.compute_descriptors.return_value = np.zeros((2, 10))

        config = SurrogateConfig(model_type="mock", force_threshold=50.0, n_samples=10)
        pipeline = SurrogatePipeline(uat_db, config, model=mock_model)
        pipeline.run()

        # Post-conditions
        # Invalid should be rejected
        assert uat_db.count(selection="status=rejected") == 1
        # Valid should be selected
        assert uat_db.count(selection="status=selected") == 1

def test_scenario_3_3_diversity(uat_db):
    """
    Scenario 3.3: Diversity Selection (FPS)
    """
    with uat_db:
        # We add 50 structures.
        for i in range(50):
            uat_db.save_candidate(Atoms('H'), metadata={"status": "pending"})

        mock_model = MagicMock()
        mock_model.compute_energy_forces.return_value = (np.zeros(50), [np.zeros((1,3))]*50)

        # Descriptors: 40 are [0,0], 10 are [10,10]
        descs = np.zeros((50, 2))
        descs[40:] = 10.0 # Last 10 are different (Liquids)

        mock_model.compute_descriptors.return_value = descs

        config = SurrogateConfig(model_type="mock", n_samples=5)
        pipeline = SurrogatePipeline(uat_db, config, model=mock_model)
        pipeline.run()

        # FPS should pick at least one from the "liquid" group (index >= 40)
        selected_atoms = uat_db.get_entries(selection="status=selected")
        selected_ids = [id for id, _ in selected_atoms]

        # IDs are 1-based index in this test context
        # 1..40 are crystals, 41..50 are liquids

        has_liquid = any(id > 40 for id in selected_ids)
        assert has_liquid, "FPS failed to select distinct structure"
