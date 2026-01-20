# Cycle 04 UAT: Surrogate Explorer

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-04-01** | High | **MACE Pre-screening** | Verify that the system utilizes the MACE foundation model to predict forces and successfully filters out structures that exhibit unphysical forces (e.g., > 100 eV/A) due to atomic overlap. |
| **UAT-04-02** | High | **Diversity Sampling (FPS)** | Verify that Farthest Point Sampling (FPS) selects a subset of structures that is geometrically more diverse than a random selection. |
| **UAT-04-03** | Medium | **Descriptor Calculation** | Verify that invariant structural fingerprints (SOAP/ACE) can be calculated for arbitrary unit cells, and that these fingerprints are invariant to rotation and translation. |
| **UAT-04-04** | Low | **Hardware Agnosticism** | Verify that the code runs on both CPU (fallback) and CUDA (if available) without crashing, adjusting batch sizes if necessary. |

### Recommended Demo
Create `demo_04_surrogate.ipynb`.
1.  **Block 1**: Generate a synthetic dataset of a diatomic molecule with bond lengths varying from 0.5A to 5.0A (100 steps).
2.  **Block 2**: Compute SOAP descriptors for all 100 structures.
3.  **Block 3**: Run FPS to select 5 structures.
4.  **Block 4**: Visualize the selected bond lengths. They should be roughly `[0.5, 5.0, 2.75, 1.6, 3.8]` (endpoints, then midpoint, then quarters).
5.  **Block 5**: Pass a "Exploded" structure to the MACE filter and show it gets rejected.
6.  **Block 6**: Show a PCA plot of the descriptor space, highlighting selected points.

## 2. Behavior Definitions

### Scenario: High Force Filtering
**GIVEN** a batch of 10 candidate structures, where one structure has two atoms overlapping (distance < 0.5A).
**WHEN** passed to `MaceClient.filter_unphysical(threshold=50.0)`.
**THEN** the overlapping structure should be excluded from the returned list.
**AND** the log should verify "Excluded 1 structure due to high forces".
**AND** the filtered list length should be 9.

### Scenario: FPS Selection Logic
**GIVEN** a pool of 100 structures where 90 are identical (Cluster A) and 10 are unique (Cluster B).
**WHEN** `FPSSampler.select(n=5)` is called.
**THEN** the algorithm should preferentially pick from Cluster B after the first pick from Cluster A, because they are "farther" away in descriptor space.
**AND** the final selection should contain a higher proportion of unique structures than a random sample would.
**AND** the returned indices should be unique.

### Scenario: Rotational Invariance
**GIVEN** a structure `A` and a structure `B` which is `A` rotated by 90 degrees.
**WHEN** descriptors are calculated for both.
**THEN** the distance `norm(desc(A) - desc(B))` should be close to zero (within numerical tolerance).
**THIS** proves that the machine learning model will see them as the same physical object.

### Scenario: Model Loading
**GIVEN** a configured `SurrogateConfig`.
**WHEN** `MaceClient` is instantiated.
**THEN** it should successfully load the model weights (or mock) without crashing.
**AND** report the device being used (CPU/CUDA).
**AND** it should be able to process a batch of atoms.
