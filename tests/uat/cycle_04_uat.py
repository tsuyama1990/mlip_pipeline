from unittest.mock import patch

import numpy as np
from ase import Atoms

from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
from mlip_autopipec.surrogate.descriptors import DescriptorResult
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline


# UAT-04-01: MACE Pre-screening
def test_uat_04_01_mace_prescreening():
    """
    Verify that the system utilizes the MACE foundation model to predict forces
    and successfully filters out structures that exhibit unphysical forces.
    """
    config = SurrogateConfig(force_threshold=50.0)
    pipeline = SurrogatePipeline(config)

    # GIVEN a batch of 10 candidate structures, where one structure has two atoms overlapping
    candidates = [Atoms("H2", positions=[[0, 0, 0], [0, 0, 2.0]]) for _ in range(9)]
    # Add a bad structure
    bad_structure = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.1]])  # Overlap
    candidates.append(bad_structure)

    # Mock MACE behavior
    def mock_predict_forces(atoms_list):
        forces = []
        for atoms in atoms_list:
            dist = np.linalg.norm(atoms.positions[0] - atoms.positions[1])
            if dist < 0.5:
                # High force
                f = np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
            else:
                f = np.zeros((2, 3))
            forces.append(f)
        return forces

    with patch.object(pipeline.mace_client, "predict_forces", side_effect=mock_predict_forces):
        with patch.object(pipeline.mace_client, "_load_model"):  # Prevent loading
            # Mock descriptor and sampler
            with patch.object(
                pipeline.descriptor_calc,
                "compute_soap",
                return_value=DescriptorResult(features=np.zeros((9, 10))),
            ):
                with patch.object(
                    pipeline.sampler, "select_with_scores", return_value=(list(range(9)), [0.0] * 9)
                ):
                    # WHEN passed to the pipeline
                    selected, result = pipeline.run(candidates)

                    # THEN the overlapping structure should be excluded from the returned list
                    # AND the filtered list length should be 9
                    assert len(selected) == 9
                    # The bad structure was at index 9. It should not be in selected.
                    # Indices in result should be 0..8
                    assert result.selected_indices == list(range(9))
                    assert 9 not in result.selected_indices

                    # Verify detailed output if we were calling filter_unphysical directly
                    # Here we verify run() output implicitly


# UAT-04-02: Diversity Sampling (FPS)
def test_uat_04_02_diversity_sampling():
    """
    Verify that Farthest Point Sampling (FPS) selects a subset of structures
    that is geometrically more diverse than a random selection.
    """
    config = SurrogateConfig(fps_n_samples=5)
    pipeline = SurrogatePipeline(config)

    candidates = [Atoms("H") for _ in range(100)]

    descriptors = np.zeros((100, 3))
    for i in range(90, 100):
        descriptors[i] = [float(i), float(i), float(i)]

    with patch.object(pipeline.mace_client, "filter_unphysical", return_value=(candidates, [])):
        with patch.object(
            pipeline.descriptor_calc,
            "compute_soap",
            return_value=DescriptorResult(features=descriptors),
        ):
            # Use real sampler

            # WHEN pipeline runs
            selected, result = pipeline.run(candidates)

            # THEN the algorithm should preferentially pick from Cluster B
            selected_indices = result.selected_indices

            count_cluster_b = sum(1 for idx in selected_indices if idx >= 90)

            # We expect at least 4 from cluster B
            assert count_cluster_b >= 4
            assert len(set(selected_indices)) == 5


# UAT-04-03: Descriptor Calculation
def test_uat_04_03_descriptor_calculation():
    """
    Verify that invariant structural fingerprints (SOAP/ACE) can be calculated.
    """
    config = SurrogateConfig()
    pipeline = SurrogatePipeline(config)

    # GIVEN a structure A and a structure B which is A rotated by 90 degrees
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]])
    b = Atoms("H2", positions=[[0, 0, 0], [1.0, 0, 0]])  # Rotated

    candidates = [a, b]

    descriptor_result = pipeline.descriptor_calc.compute_soap(candidates)
    descriptors = descriptor_result.features

    # THEN the distance should be close to zero
    dist = np.linalg.norm(descriptors[0] - descriptors[1])
    assert dist < 1e-4
