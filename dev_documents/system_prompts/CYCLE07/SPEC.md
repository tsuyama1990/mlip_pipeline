# Cycle 07 Specification: Inference & Active Learning (Module E)

## 1. Summary

Cycle 07 implements the "Explorer" capability (Module E). Now that we have a trained potential (from Cycle 06), we can run large-scale Molecular Dynamics (MD) simulations. However, since the potential is trained on limited data, it will eventually encounter a configuration it doesn't understand (e.g., a rare transition state).

We implement an **Active Learning Inference Engine** using LAMMPS. This engine monitors the "Extrapolation Grade" ($\gamma$)—a metric of uncertainty—during the simulation.
1.  **Stop-on-Uncertainty**: If $\gamma$ exceeds a threshold (e.g., 5.0), the simulation pauses immediately.
2.  **Cluster Extraction**: We identify the atom(s) with the highest uncertainty. We extract a local cluster around them.
3.  **Periodic Embedding**: Crucially, we don't just cut a vacuum cluster. We wrap it in a periodic box with a buffer zone.
4.  **Force Masking**: We verify which atoms are in the "core" (valid environment) vs "buffer" (artificial boundary). We tag the buffer atoms so their forces are ignored (masked) during the next training round.

This cycle enables the system to autonomously discover new physics.

## 2. System Architecture

Files marked in **bold** are new or modified in this cycle.

### 2.1. File Structure

```ascii
mlip_autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── config/
│       │   └── schemas/
│       │       └── **inference.py**    # Inference Configuration Schema
│       ├── **inference/**
│       │   ├── **__init__.py**
│       │   ├── **runner.py**           # LAMMPS Runner with Active Learning
│       │   └── **embedding.py**        # Cluster Extraction & Masking Logic
│       └── utils/
│           └── **lammps_writer.py**    # Helper for LAMMPS input generation
└── tests/
    └── inference/
        ├── **test_runner.py**
        └── **test_embedding.py**
```

### 2.2. Code Blueprints

#### `src/mlip_autopipec/config/schemas/inference.py`
Defines the simulation.

```python
from pydantic import BaseModel, Field

class InferenceConfig(BaseModel):
    temperature: float = Field(..., description="Simulation temperature (K)")
    pressure: float = Field(default=0.0, description="Pressure (bar)")
    n_steps: int = 1000000
    uncertainty_threshold: float = Field(default=5.0, description="Max allowed extrapolation grade")
    embedding_cutoff: float = 5.0
    embedding_buffer: float = 3.0
```

#### `src/mlip_autopipec/inference/runner.py`
Runs LAMMPS with `fix halt`.

```python
class LammpsRunner:
    def __init__(self, config: InferenceConfig, potential_path: str):
        self.config = config
        self.potential_path = potential_path

    def run_md(self, atoms: Atoms):
        """
        1. Write data file.
        2. Write input script.
           - pair_style pace
           - compute unc all pace/extrapolation_grade ...
           - fix halt ... if v_max_unc > threshold
        3. Run LAMMPS.
        4. Check if halted.
        5. If halted, return the snapshot.
        """
        pass
```

#### `src/mlip_autopipec/inference/embedding.py`
The "Surgery" module.

```python
import numpy as np
from ase import Atoms

class EmbeddingExtractor:
    def extract_cluster(self, big_atoms: Atoms, center_index: int, r_core: float, r_buffer: float) -> Atoms:
        """
        1. Identify neighbors of center_index within r_core + r_buffer.
        2. Create a new Atoms object.
        3. Set cell size to wrap the cluster (approx 2*(r_core+r_buffer)).
        4. Generate 'force_mask' array: 1.0 if dist < r_core, else 0.0.
        5. Return the small, periodic box.
        """
        # Complex logic handling PBC distances
        pass
```

## 3. Design Architecture

### 3.1. Domain Concepts

1.  **The "Fix Halt" Trick**: LAMMPS has a built-in command `fix halt` that stops the loop if a variable condition is met. We compute the max uncertainty per step. If `max_gamma > threshold`, LAMMPS exits with a specific code. This saves us from writing a slow Python loop calling LAMMPS every step.
2.  **Force Masking**: When we extract a cluster, the outer atoms ("buffer") see a fake environment (vacuum or self-image). Their forces are "wrong" (physical artifacts). We must tell the Training Engine (Cycle 06) to ignore them. We store a `force_mask` array in `atoms.arrays`. Pacemaker supports this via `weight` keywords.

### 3.2. Consumers and Producers

-   **Consumer**: `LammpsRunner` consumes the `current.yace` potential and a starting structure.
-   **Producer**: `EmbeddingExtractor` produces `PENDING` candidates (small clusters) to be sent back to Cycle 01/04 (Database/DFT).

## 4. Implementation Approach

### Step 1: LAMMPS Input Writer
-   **Task**: Implement a helper to generate the LAMMPS script.
-   **Detail**: Must correctly load the `.yace` potential using `pair_style pace`. Must set up the `compute` for extrapolation grade.

### Step 2: The Runner
-   **Task**: Implement `LammpsRunner`.
-   **Detail**: Handle the "Exit Code" from LAMMPS to distinguish between "Finished Successfully" (time up) and "Halted" (Uncertainty found).

### Step 3: Embedding Logic
-   **Task**: Implement `EmbeddingExtractor`.
-   **Detail**: This is tricky. We need to cut a sphere, but simulation cells are parallelepipeds. We define a cubic cell large enough to hold the sphere and enforce PBC. We rely on `ase.neighborlist` for neighbor finding.

## 5. Test Strategy

### 5.1. Unit Testing Approach (Min 300 words)

-   **Embedding Logic**:
    -   *Test*: Create a 1D chain of atoms. Extract a cluster around atom 5 with radius 1.
    -   *Assert*: The returned atoms contain atoms 4, 5, 6.
    -   *Assert*: The `force_mask` is `[0, 1, 0]` (Center is valid, neighbors are buffer).
    -   *Assert*: The new cell is small.
-   **Input Script**:
    -   *Test*: Generate script. Check for `fix halt` command. Check that `uncertainty_threshold` matches config.

### 5.2. Integration Testing Approach (Min 300 words)

-   **Mock LAMMPS Run**:
    -   We can't rely on `lmp_serial` being installed.
    -   *Test*: Mock the runner to return a "Halted" status and a dummy snapshot with high uncertainty.
    -   *Action*: Pass snapshot to `EmbeddingExtractor`.
    -   *Result*: A new candidate is generated.
-   **Force Mask Propagation**:
    -   *Test*: Extract a cluster. Save to DB.
    -   *Verify*: Read back from DB. Ensure `arrays['force_mask']` is preserved. (Requires DB support from Cycle 01/04).
