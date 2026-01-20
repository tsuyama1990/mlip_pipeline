# Cycle 03: Surrogate Explorer

## 1. Summary

In Cycle 03, we introduce "Intelligence" into the data collection pipeline. Traditional high-throughput materials discovery approaches often rely on "Brute Force" sampling—simulating every possible adsorption site, every alloy configuration, or every time step of an MD trajectory. This approach is computationally wasteful. A significant portion of randomly generated structures are either redundant (providing no new information) or physically nonsensical (e.g., atoms overlapping, causing expensive DFT crashes).

Cycle 03 implements **Module B: Surrogate Explorer**. This module serves as a sophisticated filter and selector placed between the Generator (Cycle 2) and the DFT Factory (Cycle 1). It leverages a pre-trained **Foundation Model**—specifically **MACE-MP**, which has been trained on the massive Materials Project database—to "audition" candidate structures before they are allowed to consume expensive DFT resources.

The workflow consists of two critical stages:
1.  **Pre-screening (The Bouncer)**: The MACE model rapidly predicts the atomic forces for thousands of candidates. Any structure exhibiting forces exceeding a safety threshold (e.g., $50 \text{ eV/\AA}$) indicates a high-energy steric clash or a broken bond. These structures are rejected instantly. This prevents the "Poison Pill" scenario where a bad structure hangs the DFT solver for hours.
2.  **Active Selection (The Scout)**: For the valid structures, we move beyond random selection. We extract the **structural descriptors** (the high-dimensional latent representation from the MACE model's penultimate layer). We then apply **Farthest Point Sampling (FPS)** to these descriptors. This algorithm mathematically guarantees that we select a subset of structures that is maximally diverse, covering the widest possible region of the chemical phase space.

By the end of this cycle, the system will be able to transform a raw pool of 10,000 noisy, redundant candidates into a refined "Gold Standard" batch of 50-100 structures that maximize information gain for the subsequent training phase.

## 2. System Architecture

### 2.1. Code Blueprint and File Structure

The surrogate module acts as a bridge. It imports data structures from `core` and `generator`, processes them, and prepares them for `dft`.

The following file structure will be implemented. Files in **bold** are the primary deliverables.

```
mlip_autopipec/
├── surrogate/
│   ├── **__init__.py**
│   ├── **config.py**           # Pydantic schemas for Model paths and Thresholds
│   ├── **models.py**           # Data structures for Surrogate Results (Scores, Descriptors)
│   ├── **mace_client.py**      # The wrapper around the PyTorch MACE model
│   ├── **fps_selector.py**     # The Farthest Point Sampling logic
│   └── **pipeline.py**         # The orchestrator function connecting MACE -> FPS
└── tests/
    └── surrogate/
        ├── **test_mace.py**
        ├── **test_fps.py**
        └── **test_pipeline.py**
```

### 2.2. Component Interaction and Data Flow

1.  **Input Stream**:
    The pipeline receives a stream (list) of `ase.Atoms` objects from the Generator. These are untagged and unvalidated.

2.  **Surrogate Evaluation (The MACE Client)**:
    The `MaceClient` accepts the list. It implements a batching strategy (e.g., chunks of 64) to optimize GPU throughput.
    -   It runs the forward pass of the MACE model.
    -   It extracts three quantities: **Potential Energy** ($E$), **Atomic Forces** ($F$), and **Invariant Descriptors** ($D$).
    -   *Optimization*: The client checks if a GPU is available. If so, it moves tensors to CUDA; otherwise, it falls back to CPU.

3.  **Filtering Logic**:
    The pipeline iterates through the results.
    -   It calculates $F_{max} = \max_i \| \mathbf{f}_i \|$ for each structure.
    -   It compares $F_{max}$ against `SurrogateConfig.force_threshold` (default 50 eV/A).
    -   Structures violating the threshold are flagged as `REJECTED`. The reason is logged (e.g., "Max Force 120.5 eV/A > 50.0").

4.  **Diversity Selection (The FPS Selector)**:
    The `FPSSampler` receives the `ACCEPTED` structures and their descriptors $D$.
    -   It computes the pairwise distance matrix (Euclidean or Cosine) between all candidates.
    -   It runs the Greedy FPS algorithm to select $K$ indices.
    -   *Input*: 1000 valid candidates. *Output*: 50 selected indices.

5.  **Output Stream**:
    The selected `Atoms` objects are updated with metadata: `info['surrogate_score']`, `info['descriptor_pca']`. They are returned to the Orchestrator to be queued for DFT.

## 3. Design Architecture

### 3.1. Surrogate Configuration (`surrogate/config.py`)

-   **`SurrogateConfig`**:
    -   `model_checkpoint`: `str` (default "medium"). Can be a local path or a URL ("https://...").
    -   `device`: `Literal["cpu", "cuda", "mps"]`.
    -   `batch_size`: `int` (default 32).
    -   `force_threshold`: `float` (default 50.0). The cutoff for "unphysical" structures.
    -   `n_selection`: `int` (default 100). How many structures to pick for DFT.

### 3.2. MACE Client (`surrogate/mace_client.py`)

This class manages the complex interaction with the `mace-torch` library and PyTorch.

-   **Class**: `MaceClient`
-   **Method**: `load_model()`:
    -   Uses `torch.load` to load the model.
    -   Sets `model.eval()` to disable dropout/gradients.
    -   Freezes parameters to save memory.
-   **Method**: `evaluate_batch(atoms_list: List[Atoms]) -> List[SurrogateResult]`
    -   **Preprocessing**: Converts ASE atoms to MACE `AtomicData` (graph representations).
    -   **Inference**: `out = model(batch)`.
    -   **Feature Extraction**: To get descriptors, we need to hook into the model. MACE models usually expose `node_feats` (the atomic environment vectors). We perform **Global Average Pooling** on these node features to get a fixed-size vector for the whole structure.
    -   **Return**: A list of objects containing the predicted properties and the pooled descriptor.

### 3.3. FPS Selector (`surrogate/fps_selector.py`)

This class implements the mathematical selection logic.

-   **Class**: `FPSSampler`
-   **Method**: `select(candidates: List[SurrogateResult], n: int) -> List[int]`
    -   **Math**: Let $S$ be the set of selected points. Initially $S = \{p_0\}$ (random or max force).
    -   **Iteration**: In each step, choose point $u \in \text{Candidates}$ that maximizes $\min_{v \in S} d(u, v)$.
    -   **Matrix Implementation**:
        1.  Maintain an array `min_dists` of size $N$ (candidates), initialized to $\infty$.
        2.  When point $s$ is added to $S$, update: `min_dists[i] = min(min_dists[i], dist(candidate[i], s))`.
        3.  Pick `argmax(min_dists)`.
    -   **Distance Metric**: We use Euclidean distance $\|\mathbf{x} - \mathbf{y}\|_2$. Since descriptors are high-dimensional, we might optionally normalize them first.

### 3.4. Surrogate Data Models (`surrogate/models.py`)

-   **`SurrogateResult`**:
    -   `uuid`: Unique ID.
    -   `energy_pred`: float.
    -   `forces_pred`: Nx3 array.
    -   `max_force`: float.
    -   `descriptor`: 1D array (vector).
    -   `is_valid`: bool.

## 4. Implementation Approach

1.  **Dependency Handling**:
    -   `mace-torch` is a heavy dependency and might not be present in all environments (e.g., CI/CD).
    -   **Strategy**: We wrap imports in `try/except ImportError`. If missing, `MaceClient` raises a clear error instructing the user to install it.
    -   **Mocking**: For unit tests, we will aggressively mock `mace-torch` so we don't need the GPU or the 100MB model file.

2.  **Model Management**:
    -   We need to handle the downloading of the foundation model (MACE-MP).
    -   **Strategy**: We check `~/.cache/mlip_autopipec/mace.pt`. If missing, we download it using `urllib` from the official repository.

3.  **FPS Efficiency**:
    -   Naive FPS is $O(KN)$ where $K$ is selection size and $N$ is pool size. With $N=10,000$, this is fast enough in Python/NumPy.
    -   If $N$ grows to $10^6$, we will need to implement a block-based approach, but for Cycle 3, NumPy is sufficient.

4.  **CLI Integration**:
    -   Add `mlip-auto select --input candidates.xyz --output selected.xyz --n 50`.

## 5. Test Strategy

### 5.1. Unit Testing
-   **FPS Logic**:
    -   Create synthetic 2D data: a cluster at (0,0) and an outlier at (10,10).
    -   Run FPS with $N=2$.
    -   Assert it picks one from the cluster and the outlier.
    -   Test Edge Case: Request $N > \text{total_candidates}$. Should raise ValueError or return all.
-   **Filter Logic**:
    -   Input: 3 structures. A (F=1.0), B (F=60.0), C (F=40.0). Threshold=50.
    -   Result: A and C accepted. B rejected.

### 5.2. Integration Testing
-   **Mock MACE Pipeline**:
    -   Mock `MaceClient` to return random forces and random descriptors (seeds fixed).
    -   Feed 100 generated structures.
    -   Run the full `pipeline.run()`.
    -   Check that we get exactly $N$ outputs.
    -   Check that the output atoms have the `surrogate_score` info tag.
-   **Real Model Test (Local)**:
    -   Requires real MACE installation.
    -   Load a water molecule.
    -   Run inference.
    -   Check that Energy is approx -75 eV (MACE scale) and not zero.

### 5.3. Performance Profiling
-   **Throughput**: Measure time to process 100 structures.
-   **Target**: The surrogate must be at least 100x faster than DFT. If DFT takes 1 hour, Surrogate batch must take < 30 seconds.
