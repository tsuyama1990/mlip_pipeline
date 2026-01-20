# Cycle 04 Specification: Surrogate Explorer

## 1. Summary

Cycle 04 implements **Module B: Surrogate Explorer**. In the previous cycle, we built a generator capable of creating thousands of structures. However, DFT calculations (Cycle 02) are expensive. We cannot calculate everything. We need a "Funnel".

The Surrogate Explorer uses a **Foundation Model** (specifically MACE-MP, pre-trained on the Materials Project) to act as a "Scout". It predicts the energy and forces of generated structures at a fraction of the cost of DFT.
1.  **Pre-Screening**: We filter out structures that are physically catastrophic (e.g., forces > 100 eV/A) which would just crash the DFT code or waste time.
2.  **Diversity Selection**: We use **Farthest Point Sampling (FPS)** based on structural fingerprints (SOAP or ACE descriptors) to select a subset of structures that covers the maximum geometric variance. This ensures that every DFT calculation adds unique information to the dataset.

By the end of this cycle, we will have a pipeline that can ingest 10,000 raw candidates and output the "Top 100" most valuable structures for labeling.

## 2. System Architecture

New components in `src/surrogate`.

```ascii
mlip_autopipec/
├── src/
│   ├── surrogate/
│   │   ├── __init__.py
│   │   ├── **mace_client.py**  # Interface to MACE-MP inference
│   │   ├── **sampling.py**     # FPS algorithm implementation
│   │   └── **descriptors.py**  # SOAP/ACE fingerprint calculation
│   └── config/
│       └── models.py           # Updated with SurrogateConfig
├── tests/
│   └── surrogate/
│       ├── **test_mace.py**
│       ├── **test_sampling.py**
│       └── **test_descriptors.py**
```

### Key Components

1.  **`MaceClient`**: A wrapper around `mace-torch`. It loads the model (lazily, to save memory) and provides a `predict(atoms_list)` method. It returns approximate Energy and Forces. It handles batching for GPU efficiency.
2.  **`DescriptorCalculator`**: Wraps `dscribe` (or uses MACE internal layers) to compute a fixed-length vector representation for each structure. We will likely use global SOAP (averaged over atoms) to get a single vector per structure.
3.  **`FPSSampler`**: Implements the Greedy Farthest Point Sampling algorithm. It takes a matrix of feature vectors and returns the indices of the $N$ most distinct samples.

## 3. Design Architecture

### Domain Concepts

**The "Scout" Strategy**:
Think of MACE-MP not as the "Truth", but as a "Compass". It might get the absolute energy wrong, but it generally knows if a structure is stable or exploding. It also knows if two structures are similar.
-   **Step 1**: Generate 10,000 candidates.
-   **Step 2**: Run MACE. Discard 1,000 "exploding" ones.
-   **Step 3**: Compute Descriptors for 9,000.
-   **Step 4**: Select 100 via FPS.
-   **Step 5**: Send 100 to DFT.

**Farthest Point Sampling (FPS)**:
Objective: Maximize the minimum distance between selected points.
$S = \{s_0\}$ (random start)
Loop $k=1 \dots N$:
  $s_k = \text{argmax}_{x \in X} (\min_{s \in S} d(x, s))$
  Add $s_k$ to $S$.
This algorithm is computationally expensive ($O(N^2)$), so we must optimize it for large $N$.

### Data Models

```python
class SurrogateConfig(BaseModel):
    model_path: str = "medium" # MACE model size
    device: Literal["cpu", "cuda"] = "cuda"
    fps_n_samples: int = 100 # How many to select
    force_threshold: float = 50.0 # eV/A, filter limit
    descriptor_type: Literal["soap", "ace", "mace_latent"] = "soap"

class SelectionResult(BaseModel):
    selected_indices: List[int]
    scores: List[float] # Distance to set
```

## 4. Implementation Approach

1.  **Step 1: Descriptor Calculation (`descriptors.py`)**:
    -   Install `dscribe`.
    -   Implement `DescriptorCalculator.compute_soap(atoms_list)`.
    -   Use `dscribe.descriptors.SOAP` with `average='inner'` to get a global descriptor.
    -   Parameters: $r_{cut}=6.0, n_{max}=8, l_{max}=6$.
2.  **Step 2: FPS Logic (`sampling.py`)**:
    -   Implement `FPSSampler.select(features, n)`.
    -   Use `scipy.spatial.distance.cdist` to compute distances.
    -   Optimization: For large sets, do not compute the full $N \times N$ matrix. Maintain a `min_dist` array for the pool and update it incrementally.
3.  **Step 3: MACE Client (`mace_client.py`)**:
    -   Implement `MaceClient`.
    -   Method `filter_unphysical(atoms_list, threshold)`.
    -   Run MACE inference. Check `max(norm(forces))`.
    -   Return filtered list and rejected list (for logging).
4.  **Step 4: Integration**:
    -   Create a pipeline function in `surrogate/pipeline.py` that ties these steps together.
    -   Ensure memory management (delete large tensors/models) after use.

## 5. Test Strategy

### Unit Testing Approach (Min 300 words)
-   **FPS Algorithm**: We will test the pure math logic. We will create points on a 1D line: `[0, 1, 2, ... 10]`. If we select 2 points, FPS *must* pick `0` and `10` (endpoints). If we select 3, it should pick `0, 10, 5`. We will verify this deterministic behavior. We will also test with random points to ensure it doesn't crash.
-   **Descriptor Shape**: We will pass a batch of 5 structures to `DescriptorCalculator`. We will assert that the output is a Numpy array of shape `(5, n_features)`. We will check that rotating the molecule (without changing shape) results in an identical (or very close) SOAP descriptor (rotational invariance).
-   **MACE Filter**: We will create a "Bad" atom with two atoms at distance 0.1A. We will mock the MACE return value to give `Force = 1000 eV/A`. We will verify that `filter_unphysical` correctly removes this structure from the list. We will also verify it keeps the "Good" structures.

### Integration Testing Approach (Min 300 words)
-   **Mock MACE**: MACE is a large model. For CI/Integration tests, we will not download the 500MB weights. We will use a `MockMaceClient` that returns random (but reproducible) forces.
-   **Full Pipeline**:
    1.  Generate 50 random structures (Cycle 03 code).
    2.  Pass them to the Surrogate Pipeline.
    3.  Mock the Descriptor calculation (return random vectors).
    4.  Request 5 samples.
    5.  Verify that 5 structures are returned.
    6.  Verify they are distinct (check IDs).
    7.  Verify the rejected ones are logged.
-   **Performance Check**: We will measure the time it takes to compute descriptors and run FPS on a set of 1000 structures. It should be in the order of seconds, not minutes. This ensures our `cdist` implementation is efficient and vectorized.
