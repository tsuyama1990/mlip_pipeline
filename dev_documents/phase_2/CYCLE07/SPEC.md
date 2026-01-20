# Cycle 07 Specification: Scalable Inference Engine (Part 2)

## 1. Summary

Cycle 07 completes **Module E: Scalable Inference Engine**. In Cycle 06, we detected "Uncertain Structures". These structures are often large supercells (e.g., 1000 atoms) from an MD simulation. We cannot afford to run DFT on the entire supercell just to fix a local error.

This cycle implements the **Periodic Embedding & Force Masking** strategy.
1.  **Extraction**: We identify the "focal atom" (the one with high uncertainty). We cut out a local cluster around it (e.g., radius 6A).
2.  **Embedding**: We do not treat this cluster as an isolated molecule (which would create surface effects). We treat it as a small periodic system.
3.  **Force Masking**: The atoms at the edge of this new small box are in an artificial environment. Their DFT forces will be "wrong" (dominated by the cut). We calculate a **Mask**.
    -   Inner Core (Radius < 4A): Force Weight = 1.0 (Trusted).
    -   Buffer Shell (4A < Radius < 6A): Force Weight = 0.0 (Ignored).
    This allows us to train the potential on the local environment without polluting the dataset with surface artifacts.

By the end of this cycle, we will have a robust method to turn "Bad MD Frames" into "Good Training Data".

## 2. System Architecture

New components in `src/inference`.

```ascii
mlip_autopipec/
├── src/
│   ├── inference/
│   │   ├── ...
│   │   ├── **embedding.py**    # Logic for excising clusters
│   │   └── **masking.py**      # Logic for calculating force weights
│   └── config/
│       └── models.py           # Updated
├── tests/
│   └── inference/
│       ├── **test_embedding.py**
│       └── **test_masking.py**
```

### Key Components

1.  **`EmbeddingExtractor`**: Takes a large `Atoms` object and a list of indices (the uncertain atoms). It returns a list of smaller `Atoms` objects (the embeddings). It handles the Periodic Boundary Conditions (PBC) logic to ensure correct neighbor finding across boundaries.
2.  **`ForceMasker`**: Adds an array `force_mask` (and optionally `energy_weight`) to the `Atoms.info/arrays`. This will be read by the `DatasetBuilder` in Cycle 05 to set weights in the training file.

## 3. Design Architecture

### Domain Concepts

**The "Cluster-in-Box" Approach**:
We assume that the interaction range of the potential is finite (cutoff $R_c$). If we cut a sphere of radius $R_{cut} + R_{buffer}$, the forces on the central atoms are physically identical to the forces in the bulk, up to the accuracy of the buffer.
-   **Periodic Boundary**: We place the cluster in a cubic box of size $L = 2(R_{cut} + R_{buffer})$. We assume periodicity. This is an approximation, but better than vacuum.

**Masking**:
To the ML training code (Pacemaker), a weight of 0.0 means "Don't calculate the loss for this atom". This is crucial. We only pay for the DFT of the whole box, but we only "consume" the information from the center.

**Queue Integration**:
These extracted structures are not trained on immediately. They are added to the `DFT_Queue` with a high priority tag.

### Data Models

```python
class EmbeddingConfig(BaseModel):
    core_radius: float = 4.0
    buffer_width: float = 2.0
    # derived: box_size ~ 12.0

class ExtractedStructure(BaseModel):
    atoms: Any # ase.Atoms
    origin_uuid: str
    origin_index: int
    mask_radius: float
```

## 4. Implementation Approach

1.  **Step 1: Neighbor Search (`embedding.py`)**:
    -   Implement `EmbeddingExtractor.extract(large_atoms, center_idx)`.
    -   Use `ase.neighborlist.NeighborList` to find all atoms within $R_{core} + R_{buffer}$ of the focal atom.
    -   Collect these atoms.
2.  **Step 2: Box Construction**:
    -   Create a new `Atoms` object.
    -   Center the focal atom at $(L/2, L/2, L/2)$.
    -   Wrap positions using Minimum Image Convention.
    -   Set `pbc=True`.
3.  **Step 3: Masking (`masking.py`)**:
    -   Implement `ForceMasker.apply(atoms, center, radius)`.
    -   Compute distance of every atom from center.
    -   If $d < R_{core}$: mask = 1.
    -   Else: mask = 0.
    -   Store in `atoms.arrays['force_mask']`.

## 5. Test Strategy

### Unit Testing Approach (Min 300 words)
-   **Extraction Geometry**: We will create a perfect FCC crystal (10x10x10). We will select the central atom. We will extract a cluster with radius equal to the nearest neighbor distance. We will assert that the extracted cluster contains exactly 13 atoms (1 center + 12 neighbors). We will verify distances are preserved.
-   **Mask Values**: We will verify that the mask array is boolean (or float 0.0/1.0) and matches the geometric criteria. We will check boundary cases (atoms exactly on the radius). We will assert that the array length matches the number of atoms.
-   **PBC Handling**: We will test extraction on an atom close to the boundary of the original large supercell. We must ensure that the extractor correctly handles the wrapping (mic) to find neighbors across the periodic boundary of the source cell. If an atom is at $x=0.1$ and its neighbor is at $x=9.9$ (in a box of 10), the distance should be $0.2$, not $9.8$.

### Integration Testing Approach (Min 300 words)
-   **Cycle 2 Handoff**: We will take an extracted, masked structure and pass it to the `QERunner` (mock). We verify that `QERunner` runs (it ignores the mask, which is fine). This ensures the `Atoms` object is valid.
-   **Cycle 5 Handoff**: We will take the result from `QERunner` (now with Forces), re-apply the mask (or ensure it persisted), and pass it to `DatasetBuilder`. We verify that `DatasetBuilder` writes the weights correctly to the Pacemaker training file. This is the critical integration point: Cycle 7 -> Cycle 2 -> Cycle 5.
-   **Visual Inspection (Manual)**: We will write a test that dumps the extracted structure to `.xyz`. In a real scenario, we would visualize this to ensure the "Cluster" looks like a sphere in a box, not a fragmented cloud.
