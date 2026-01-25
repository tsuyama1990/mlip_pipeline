# Cycle 03 Specification: Surrogate Selection

## 1. Summary

Cycle 03 introduces **Module B: Surrogate Explorer**. In the previous cycle, we built a generator capable of creating thousands of physically diverse structures. However, this creates a new problem: **Combinatorial Explosion**. We cannot afford to run Density Functional Theory (DFT) on every generated structure, as DFT scales as $O(N^3)$ and consumes significant HPC resources. We face an "Information vs. Cost" trade-off: we want to simulate enough structures to learn the physics, but we can't afford to simulate everything.

This cycle implements a **"Surrogate-First"** strategy. We utilize a pre-trained foundation model (specifically **MACE-MP**, trained on the massive Materials Project dataset) as a "Scout". This model acts as a cheap proxy for DFT (orders of magnitude faster). It allows us to:
1.  **Filter Unphysical Structures**: The generator might accidentally place atoms too close together (clashing), leading to extremely high energies. MACE will predict enormous forces for these. We can discard them immediately ("Fail Fast"), preventing wasted DFT cycles.
2.  **Select Diverse Candidates**: We use **Farthest Point Sampling (FPS)** on the local atomic descriptors (fingerprints) to select a subset of structures that covers the widest possible area of the configuration space. This ensures we don't waste DFT cycles on 100 almost-identical structures (redundancy).

By the end of this cycle, the system will act as a sophisticated funnel: taking 10,000 generated structures, filtering out the "garbage", and intelligently selecting the "best" 100 for actual DFT calculation.

## 2. System Architecture

The architecture relies on the `surrogate` package, which interfaces with external ML frameworks (PyTorch/MACE) and interacts with the database.

### File Structure
**bold** files are to be created or modified.

```
mlip_autopipec/
├── surrogate/
│   ├── **__init__.py**
│   ├── **pipeline.py**         # Main Pipeline Orchestrator
│   ├── **model_interface.py**  # Abstract Base Class for Surrogates
│   ├── **mace_wrapper.py**     # MACE Implementation
│   └── **sampling.py**         # FPS Implementation
├── config/
│   └── schemas/
│       └── **surrogate.py**    # Config (Model path, batch size)
```

### Data Dictionary

| Model Name | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| **SurrogateConfig** | model_type | str | "mace_mp", "chgnet", or "mock". |
| | model_path | str | Path to weights or HuggingFace ID. |
| | device | str | "cuda" or "cpu". |
| | force_threshold | float | Max allowed force (eV/A) before rejection. |
| | n_samples | int | Number of structures to select via FPS. |
| **PipelineStats** | total_candidates | int | Number of input structures. |
| | rejected_count | int | Number of structures filtered out. |
| | selected_count | int | Number of structures selected for DFT. |

## 3. Design Architecture

### Surrogate Interface (`ModelInterface`)
We define a protocol to decouple the specific ML model from the pipeline.
-   `load_model(model_path: str, device: str)`: Loads weights.
-   `compute_energy_forces(atoms: List[Atoms]) -> Tuple[np.ndarray, np.ndarray]`: Returns Energy (eV) and Forces (eV/A) for a batch.
-   `compute_descriptors(atoms: List[Atoms]) -> np.ndarray`: Returns global or local descriptors (vectors) for sampling.

### MACE Wrapper (`MaceWrapper`)
This class adapts the `mace-torch` library.
-   **Batching**: MACE is fastest when processing multiple structures at once. The wrapper must chunk the input list of atoms (e.g., batch size 32) to utilize GPU parallelism efficiently.
-   **Descriptors**: For sampling, we need a vector representation of each structure. We can extract the features from the penultimate layer of the neural network (read-out phase) or use the `dscribe` library to compute SOAP vectors if internal MACE features are hard to access in the current version.

### Farthest Point Sampling (FPS) Logic
We need to select $K$ items from a set of $N$ candidates.
-   **Metric**: Euclidean distance in the descriptor space. $d(x, y) = ||x - y||_2$.
-   **Algorithm Steps**:
    1.  **Featurization**: Convert all $N$ structures into vectors $X \in \mathbb{R}^{N \times D}$.
    2.  **Initialization**: Pick the first structure index $i_0$ (e.g., the one with the lowest predicted energy). Initialize the `Selected` set $S = \{i_0\}$.
    3.  **Distance Calculation**: Maintain a distance array `min_dists` of size $N$, initialized to infinity.
    4.  **Loop**: While $|S| < K$:
        a. Let $last$ be the index most recently added to $S$.
        b. Update `min_dists`: for all $j$, $d_{new} = ||X_j - X_{last}||$. Update `min_dists[j] = min(min_dists[j], d_{new})`.
        c. Find index $j^*$ that maximizes `min_dists`.
        d. Add $j^*$ to $S$.
    5.  **Output**: Return the set $S$.
-   **Complexity**: This is $O(K \cdot N)$. For $N=10,000$ and $K=100$, this is computationally feasible on a CPU.

### Pipeline Logic (`SurrogatePipeline`)
1.  **Fetch**: Load all `status=PENDING` candidates from DB.
2.  **Pre-screen**: Run MACE to predict Energy and Forces.
    -   **Filter**: If $F_{max} > \text{force_threshold}$ (e.g., 50 eV/A), mark as `REJECTED` in DB. These are likely unphysical overlaps.
3.  **Featurize**: Compute descriptors for the survivors.
4.  **Sample**: Run FPS to pick top $N$ candidates.
5.  **Update**:
    -   Mark selected IDs as `SELECTED` (ready for DFT).
    -   Mark non-selected (but valid) IDs as `HELD` (backlog for future).
    -   Store the MACE-predicted energy/forces in the metadata (useful for comparison later).

## 4. Implementation Approach

1.  **Implement MACE Wrapper**: Create `surrogate/mace_wrapper.py`.
    -   Use `mace.calculators.mace_mp` if available.
    -   Implement a fallback/mock for environments without GPU or `mace-torch` (for CI/CD).
    -   Ensure it handles the `device` (cpu/cuda) correctly.
2.  **Implement FPS**: Create `surrogate/sampling.py`.
    -   Use `scipy.spatial.distance.cdist` to compute distance matrices efficiently.
    -   Implement the greedy loop.
3.  **Implement Pipeline**: Create `surrogate/pipeline.py`.
    -   Connect DB -> MACE -> Filter -> FPS -> DB.
    -   Add logging to track how many structures were rejected vs selected.
4.  **CLI**: Add `mlip-auto select` command.
    -   Arguments: `--n-samples`, `--model`.

## 5. Test Strategy

### Unit Testing
-   **FPS Algorithm**:
    -   Create 4 points in 2D: (0,0), (1,1), (0.1, 0.1), (10,10).
    -   Ask for 2 points.
    -   FPS should pick (0,0) (if first) and (10,10) (farthest), skipping (0.1, 0.1) as it's redundant.
-   **High Force Filter**:
    -   Create a dimer with $r=0.1$ Angstrom.
    -   Mock the MACE wrapper to return Force = 1000 eV/A.
    -   Pipeline filter method should return False.

### Integration Testing
-   **Selection Flow**:
    -   Populate DB with 50 structures (Cycle 02).
    -   Run `mlip-auto select --n 10`.
    -   Verify DB has exactly 10 `SELECTED` items.
    -   Verify that the "Clashed" structure (if manually inserted) is `REJECTED`.
    -   Verify metadata: `mace_energy` key exists in the selected structures.
-   **MACE Loading**:
    -   Verify that the system can actually load the MACE model (downloading from cache/web if needed) without crashing.
