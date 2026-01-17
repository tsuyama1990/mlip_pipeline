# Cycle 04: Surrogate Explorer

## 1. Summary

Cycle 04 implements **Module B: Surrogate Explorer**. This module represents the "Efficiency" in the system's goals. Running DFT on every single generated candidate (from Module A) is wasteful, as many candidates will be physically redundant (too similar) or physically impossible (atoms overlapping, creating singularities).

This module solves this by placing a fast "Surrogate Model" (MACE-MP) in front of the DFT factory. MACE-MP is a foundation model trained on the Materials Project; while not accurate enough for final properties, it is excellent at identifying "reasonable" structures. We use it to filter out garbage.

Furthermore, we implement **Active Learning Selection**. We compute structural fingerprints (descriptors) for all valid candidates and use **Farthest Point Sampling (FPS)** to select a subset that maximizes information gain. This ensures that if we have budget for 100 DFT calculations, we spend it on the 100 most distinct structures, rather than 100 variations of the same ground state.

## 2. System Architecture

We add the `surrogate` package.

```ascii
mlip_autopipec/
├── config/
├── core/
├── generators/
├── surrogate/
│   ├── __init__.py
│   ├── mace_client.py      # The fast evaluator.
│   ├── descriptors.py      # The fingerprint calculator.
│   └── fps_selector.py     # The diversity algorithm.
└── tests/
    ├── test_surrogate.py   # Validates model loading and inference.
    └── test_fps.py         # Validates selection logic.
```

### 2.1 Code Blueprints

This section details the logic for high-throughput screening.

#### 2.1.1 MACE Client (`surrogate/mace_client.py`)

This class wraps the MACE PyTorch model.

**Class `MaceClient`**
*   **Attributes**:
    *   `model` (`torch.nn.Module`): The loaded MACE model.
    *   `device` (`str`): "cuda" or "cpu".
    *   `batch_size` (`int`): Default 32.
*   **Methods**:
    *   `__init__(self, model_checkpoint: str = "medium", device: str = None)`:
        *   Loads `mace_mp` model from cache or downloads it.
        *   Moves model to device.
    *   `evaluate(self, atoms_list: List[Atoms]) -> List[Atoms]`:
        *   **Description**: Computes Energy and Forces.
        *   **Logic**:
            1.  Convert `atoms_list` to MACE `AtomicData` objects.
            2.  Batch them using `torch_geometric.data.Batch`.
            3.  Run `self.model(batch)`.
            4.  Unpack results and attach `SinglePointCalculator` to each `Atoms` object.
            5.  Return the annotated list.
    *   `filter_valid(self, atoms_list: List[Atoms], f_max_threshold: float = 100.0) -> List[Atoms]`:
        *   Removes structures where $\max|F| > f_{max}$.
        *   Removes structures with overlapping atoms (distance < 0.5A).
        *   Logs how many were removed.

#### 2.1.2 Descriptor Calculator (`surrogate/descriptors.py`)

Computes fingerprints for diversity selection.

**Class `DescriptorCalculator`**
*   **Attributes**:
    *   `method` (`str`): "soap" or "ace".
*   **Methods**:
    *   `compute(self, atoms_list: List[Atoms]) -> np.ndarray`:
        *   If `method == "soap"`:
            *   Initialize `dscribe.descriptors.SOAP`.
            *   Params: `rcut=6.0`, `nmax=8`, `lmax=6`.
            *   Call `soap.create(atoms_list)`.
            *   Return matrix `(N_samples, N_features)`.
        *   If `method == "ace"`:
            *   Use MACE model's internal representations (invariant features) if accessible.

#### 2.1.3 FPS Selector (`surrogate/fps_selector.py`)

Implements the selection algorithm.

**Class `FPSSelector`**
*   **Methods**:
    *   `select(self, descriptors: np.ndarray, n_to_select: int, existing_descriptors: Optional[np.ndarray] = None) -> List[int]`:
        *   **Input**: `descriptors` (Candidate pool), `existing_descriptors` (Already in DB).
        *   **Algorithm (Maximin)**:
            1.  Initialize distance array `min_dists`.
                *   If `existing` provided: `min_dists = min(dist(candidates, existing))`.
                *   Else: Pick random first point, calculate dists to it.
            2.  Loop `k` from 1 to `n_to_select`:
                *   Idx `next` = `argmax(min_dists)`.
                *   Add `next` to selection.
                *   Update `min_dists`: `new_dists = dist(candidates, candidate[next])`.
                *   `min_dists = minimum(min_dists, new_dists)`.
            3.  Return list of selected indices.
        *   **Optimization**: Use `scipy.spatial.distance.cdist` for bulk distance calculation, or simple numpy broadcasting if N is small (< 10k).

#### 2.1.4 Data Flow Diagram (Cycle 04)

```mermaid
graph TD
    Generators[Module A] -->|Raw Candidates| Mace[MaceClient]
    Mace -->|Evaluate| Mace
    Mace -->|Filter (F_max > 100)| Valid[Valid Candidates]

    Valid -->|Compute SOAP| Descriptors[Descriptor Matrix]
    Descriptors -->|FPS Algorithm| Selector[FPSSelector]
    Selector -->|Indices| Selected[Selected Candidates]

    Selected --> DFT[DFT Queue (Module C)]
```

## 3. Design Architecture

### 3.1 Surrogate as a Gatekeeper

The design explicitly treats the Surrogate (MACE) as a **Gatekeeper**.
*   **Cost**: DFT cost is $O(N^3)$. MACE cost is $O(N)$.
*   **Ratio**: We can afford to generate 10,000 candidates, filter them down to 5,000 valid ones, and select the best 100 for DFT. This gives us a 100x effective speedup in exploring configuration space compared to running DFT on everything.
*   **Independence**: The surrogate model (MACE-MP) is *not* the model we are training (ACE). It is a "Transfer Learning" source. We use its general knowledge of chemistry to guide our specific learning of the target material.

### 3.2 Descriptor Abstraction

We abstract the descriptor calculation.
*   **Why**: FPS depends on the metric space. "Distance" in SOAP space is different from "Distance" in ACE space.
*   **Default**: We use SOAP (Smooth Overlap of Atomic Positions) via `dscribe` because it is a robust, widely accepted standard for structural similarity.
*   **Fallback**: If `dscribe` is heavy, we can implement a simple "Pair Distribution Function" descriptor using `numpy.histogram` of pairwise distances.

### 3.3 FPS Algorithm Efficiency

Naive FPS is $O(N \cdot K \cdot D)$. For $N=10^4, K=10^3, D=10^3$, this is $10^{10}$ ops, which is fine for Python/Numpy.
*   **Memory**: Storing the distance matrix ($N \times N$) is bad ($10^4 \times 10^4 \times 8$ bytes $\approx 800$ MB).
*   **Solution**: We do **not** precompute the full distance matrix. We update `min_dists` vector ($N \times 1$) incrementally. This keeps memory usage low ($O(N)$).

## 4. Implementation Approach

1.  **Environment Check**:
    *   Check if `mace-torch` and `dscribe` are installed.
    *   If not, issue a warning and use a "RandomSelector" (mock) to allow development to proceed.

2.  **MACE Integration**:
    *   Implement `MaceClient`.
    *   Download the model file (`2023-12-03-mace-128-L1_epoch-199.model`) to a cache dir `~/.cache/mace`.
    *   Test inference on a single water molecule.

3.  **FPS Implementation**:
    *   Implement `fps_selector.py`.
    *   Use `numpy` broadcasting for the distance calculation `np.linalg.norm(X - selected, axis=1)`.
    *   Verify against a known 2D dataset (e.g., points on a grid).

4.  **Pipeline Construction**:
    *   Create `SelectionPipeline` class in `__init__.py` that stitches these steps together.

## 5. Test Strategy

### 5.1 Unit Testing

*   **FPS Logic**:
    *   Create 4 points at $(0,0), (0,1), (1,0), (1,1)$ and a cluster of points near $(0.5, 0.5)$.
    *   Request 4 points via FPS.
    *   Assert that the 4 corner points are selected, as they maximize the spread.

*   **MACE Wrapper**:
    *   Mock `mace_torch` model.
    *   Pass a list of atoms.
    *   Assert that the returned atoms have `calculator` attached and results populated.

### 5.2 Integration Testing

*   **Full Selection Loop**:
    *   Generate 100 random Al structures (Cycle 03).
    *   Run them through `SelectionPipeline`.
    *   Ask for top 10.
    *   Verify we get exactly 10 structures.
    *   Verify that none of the "exploded" structures (manually inserted for the test) are in the final set.
