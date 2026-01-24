# Cycle 03 Specification: The Surrogate Explorer (Module B)

## 1. Summary

Cycle 03 introduces intelligence into the pipeline. In the previous cycle, we built a generator capable of churning out thousands of structures. However, sending all these blindly to a DFT engine (Cycle 04) would be a waste of computational resources. Many generated structures might be "junk" (physically nonsensical) or redundant (too similar to each other).

To solve this, we implement **Module B: The Surrogate Explorer**. This module acts as a highly efficient scout. It uses a pre-trained foundation model (specifically **MACE-MP-0**, trained on the Materials Project) to evaluate the generated candidates *before* they touch the expensive DFT queue.

This cycle delivers two critical capabilities:
1.  **Direct Sampling (Screening)**: The `MaceWrapper` runs inference on thousands of candidates in seconds. It predicts energy and forces. If a structure has absurdly high forces (indicating atomic overlap not caught by simple distance checks) or high energy, it is discarded immediately.
2.  **Diversity Selection (FPS)**: We don't just want valid structures; we want *diverse* ones. We implement **Farthest Point Sampling (FPS)** based on the MACE internal descriptors (or SOAP descriptors). This algorithm selects a subset of structures that covers the widest possible area of the configuration space, ensuring that our training data is information-rich.

By the end of Cycle 03, the pipeline will produce a refined "Golden Set" of candidates, reducing the downstream DFT workload by 90% while retaining 99% of the information value.

## 2. System Architecture

Files marked in **bold** are new or modified in this cycle.

### 2.1. File Structure

```ascii
mlip_autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── config/
│       │   ├── models.py               # Updated with SurrogateConfig
│       │   └── schemas/
│       │       └── **surrogate.py**    # Surrogate Configuration Schema
│       ├── **surrogate/**
│       │   ├── **__init__.py**
│       │   ├── **mace_wrapper.py**     # Wrapper for MACE-MP model
│       │   └── **sampling.py**         # FPS Logic
│       └── utils/
│           └── **descriptors.py**      # Helper for descriptor calculation (optional)
└── tests/
    └── surrogate/
        ├── **test_mace.py**
        └── **test_sampling.py**
```

### 2.2. Code Blueprints

#### `src/mlip_autopipec/config/schemas/surrogate.py`
Configuration for the model path and selection strategy.

```python
from pydantic import BaseModel, Field
from typing import Literal

class SurrogateConfig(BaseModel):
    model_path: str = Field(default="medium", description="MACE model size/path (small, medium, large)")
    device: Literal['cpu', 'cuda'] = 'cpu'
    selection_method: Literal['random', 'fps'] = 'fps'
    selection_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="Fraction of structures to select for DFT")
    energy_threshold: float = Field(default=10.0, description="Max allowed energy/atom (eV) above ground state")
```

#### `src/mlip_autopipec/surrogate/mace_wrapper.py`
Interface to the `mace-torch` library.

```python
import torch
from ase import Atoms
from typing import List
# mace imports would go here

class MaceWrapper:
    def __init__(self, config: SurrogateConfig):
        self.config = config
        self.model = self._load_model()

    def _load_model(self):
        # Logic to load MACE-MP-0 model
        pass

    def predict(self, atoms_list: List[Atoms]) -> List[Atoms]:
        """Runs batch inference. Attaches 'mace_energy' and 'mace_forces' to info/arrays."""
        # Convert to MACE input format
        # Run model
        # Update atoms object
        return atoms_list

    def get_descriptors(self, atoms_list: List[Atoms]):
        """Extracts the final layer features for FPS."""
        pass
```

#### `src/mlip_autopipec/surrogate/sampling.py`
Implements the diversity selection logic.

```python
import numpy as np

class FarthestPointSampling:
    def select(self, descriptors: np.ndarray, n_samples: int) -> List[int]:
        """
        Selects n_samples indices from the descriptors array such that
        the distance to the nearest selected point is maximized.
        """
        n_total = descriptors.shape[0]
        selected_indices = []
        # Initial selection (can be random or lowest energy)
        first = np.random.randint(0, n_total)
        selected_indices.append(first)

        # Distance cache
        min_dists = np.full(n_total, np.inf)

        for _ in range(1, n_samples):
            # Update distances based on the last selected point
            # Pick the point with the maximum minimum distance
            pass

        return selected_indices
```

## 3. Design Architecture

### 3.1. Domain Concepts

1.  **The "Scout" Concept**: The Surrogate is not the final authority. Its predictions don't need to be perfect; they just need to be correlated with the truth. If MACE says a structure is "high energy", it probably is. If it says two structures are "identical", they probably are.
2.  **Latent Space Navigation**: We define "diversity" not in XYZ coordinate space (which is degenerate under rotation/translation) but in "Descriptor Space". MACE's internal representation (invariant equivariant features) provides a robust metric for structural similarity.
3.  **Batch Processing**: MACE and PyTorch are optimised for batches. The `MaceWrapper` must accept `List[Atoms]`, not single `Atoms`, to exploit GPU parallelism.

### 3.2. Consumers and Producers

-   **Consumer**: `MaceWrapper` consumes the list of candidates produced by the `Generator` (Cycle 02).
-   **Producer**: `Sampling` produces a filtered list of indices/atoms.
-   **Integration point**: The `WorkflowManager` (Cycle 08) will pipe `Generator -> MaceWrapper -> Sampling -> Database`.

## 4. Implementation Approach

### Step 1: MACE Integration
We need to handle the dependency on `mace-torch`.
-   **Task**: Implement `MaceWrapper`. It should load the model (downloading from cache if needed) and run inference.
-   **Constraint**: If `mace-torch` is not installed (e.g., in some CI environments), the wrapper should degrade gracefully or provide a "mock" mode for testing.

### Step 2: Farthest Point Sampling (FPS)
This is a pure algorithmic step.
-   **Task**: Implement the FPS algorithm. It relies on a distance matrix (Euclidean distance in descriptor space).
-   **Optimization**: For N=10,000 structures, an $N \times N$ matrix is 100 million entries (800MB), which is manageable. For larger N, we might need a greedy approximation or batching.

### Step 3: Filtering Logic
-   **Task**: Implement `SurrogateFilter` class that combines MACE and FPS.
    -   1. Run MACE.
    -   2. Filter by Energy Threshold (remove outliers).
    -   3. Filter by Force Threshold (remove explosions).
    -   4. Compute Descriptors.
    -   5. Run FPS to select top K%.

## 5. Test Strategy

### 5.1. Unit Testing Approach (Min 300 words)

-   **Wrapper Mocking**: Since MACE models are large (hundreds of MB), unit tests should not download them. We will mock the `mace.calculators.MACECalculator` class.
    -   *Test*: Pass a list of 5 atoms. Mock returns 5 energies and $5 \times N \times 3$ forces. Verify the wrapper correctly assigns these to `atoms.info`.
-   **FPS Algorithm**: This is easy to test with synthetic data.
    -   *Test*: Create 2D points: three in a cluster, one far away.
    -   *Action*: Request 2 points via FPS.
    -   *Expectation*: The algorithm should pick one from the cluster and the far-away one. It should *not* pick two from the cluster.
    -   *Edge Case*: Request more points than available (should raise error or return all).
-   **Threshold Logic**:
    -   *Test*: Pass atoms with energies [0.0, 5.0, 100.0]. Threshold = 10.0.
    -   *Expectation*: The atom with 100.0 eV is dropped.

### 5.2. Integration Testing Approach (Min 300 words)

-   **Real Model Test (Optional/Slow)**: If the environment permits (GPU available), run the real MACE-MP-0 model on a small molecule (H2O).
    -   *Verification*: Energies are roughly reasonable (negative values). Forces are close to zero for equilibrium, high for compressed.
-   **Pipeline Integration**:
    -   Generate 50 random structures (Cycle 02 code).
    -   Pass them through `SurrogateFilter` configured to keep 10%.
    -   Assert that exactly 5 structures are returned.
    -   Assert that the returned structures are those with the most distinct descriptors (if mock descriptors are used).
-   **Performance Profiling**:
    -   Measure time taken for 1000 structures. FPS can be $O(N^2)$. Ensure it finishes in < 1 minute for typical batch sizes.
