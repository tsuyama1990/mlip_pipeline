# Cycle 02 Specification: The Physics-Informed Generator (Module A)

## 1. Summary

Cycle 02 focuses on the "Big Bang" of the MLIP-AutoPipe universe: **Structure Generation**. Before we can train a potential, we need data. And not just any data—we need physically relevant, diverse, and information-rich configurations. The goal of this cycle is to implement Module A (Physics-Informed Generator), which transforms the user's simple input (e.g., "Fe-Ni FCC") into thousands of candidate structures ready for evaluation.

We will implement three distinct generation strategies:
1.  **Combinatorial Generation (SQS)**: For alloy systems, we cannot simulate every possible arrangement of atoms. We will use Special Quasirandom Structures (SQS) to generate small supercells that statistically mimic the random correlations of an infinite solid solution.
2.  **Elastic & Thermal Perturbations**: A static crystal teaches the potential nothing about forces. We will implement a `StrainGenerator` to apply volumetric and shear strains (simulating high pressure and stress) and a `RattleGenerator` to apply Gaussian noise to atomic positions (simulating thermal vibrations). This ensures the potential learns the equation of state (EOS) and phonon properties.
3.  **Defect Engineering**: Real materials have defects. We will implement a `DefectGenerator` that automatically injects vacancies, interstitials, and antisite defects into the supercells. This is crucial for training potentials that can accurately predict diffusion barriers and mechanical failure.

By the end of this cycle, the system will be able to populate the database with a "Cold Start" dataset—thousands of `PENDING` structures—without requiring any DFT calculations yet.

## 2. System Architecture

Files marked in **bold** are new or modified in this cycle.

### 2.1. File Structure

```ascii
mlip_autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── config/
│       │   ├── models.py               # Updated with GeneratorConfig
│       │   └── schemas/
│       │       └── **generator.py**    # Generator Configuration Schema
│       ├── **generator/**
│       │   ├── **__init__.py**
│       │   ├── **builder.py**          # Base Builder & SQS Logic
│       │   ├── **transformations.py**  # Strain & Rattle Logic
│       │   └── **defects.py**          # Point Defect Injection
│       └── data_models/
│           └── status.py               # JobStatus
└── tests/
    └── generator/
        ├── **test_builder.py**
        ├── **test_transformations.py**
        └── **test_defects.py**
```

### 2.2. Code Blueprints

#### `src/mlip_autopipec/config/schemas/generator.py`
Defines how many structures to generate and what distortions to apply.

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class GeneratorConfig(BaseModel):
    supercell_size: List[int] = Field(default=[2, 2, 2], description="Supercell expansion matrix diagonal")
    num_sqs: int = Field(default=5, description="Number of distinct SQS to generate per composition")
    strain_range: tuple[float, float] = (-0.05, 0.05)
    rattle_amplitude: float = 0.1
    include_defects: bool = True
```

#### `src/mlip_autopipec/generator/transformations.py`
Applies physical distortions to atoms.

```python
import numpy as np
from ase import Atoms
from copy import deepcopy

def apply_strain(atoms: Atoms, strain_tensor: np.ndarray) -> Atoms:
    """Applies a deformation gradient to the cell and positions."""
    new_atoms = deepcopy(atoms)
    deformation = np.eye(3) + strain_tensor
    new_cell = np.dot(new_atoms.cell, deformation)
    new_atoms.set_cell(new_cell, scale_atoms=True)
    return new_atoms

def apply_rattle(atoms: Atoms, stdev: float, rng: np.random.Generator) -> Atoms:
    """Adds Gaussian noise to atomic positions."""
    new_atoms = deepcopy(atoms)
    noise = rng.normal(0, stdev, size=new_atoms.positions.shape)
    new_atoms.positions += noise
    return new_atoms
```

#### `src/mlip_autopipec/generator/defects.py`
Injects point defects using Pymatgen (via adapter) or direct ASE manipulation.

```python
from ase import Atoms
from typing import List

class DefectGenerator:
    def generate_vacancies(self, atoms: Atoms) -> List[Atoms]:
        """Returns a list of atoms objects, each with one atom removed."""
        candidates = []
        for i in range(len(atoms)):
            temp = atoms.copy()
            del temp[i]
            candidates.append(temp)
        return candidates # In reality, we filter by symmetry
```

## 3. Design Architecture

### 3.1. Domain Concepts

1.  **The "Seed" Structure**: Generation always starts from a "primitive cell" (e.g., a 2-atom FCC unit cell provided by the user).
2.  **The "Supercell" Expansion**: To capture disorder and defects, we must expand the primitive cell. A $2 \times 2 \times 2$ expansion of a 4-atom cubic cell yields 32 atoms. This is the "Working Unit" for DFT.
    -   *Constraint*: The supercell must be large enough to avoid defect self-interaction (images seeing each other), but small enough to be computationally feasible ($< 200$ atoms usually).
3.  **Compositional Enumeration**: For an alloy $A_x B_{1-x}$, we don't just generate $x$. We generate $x \pm \delta$.
    -   *Design Choice*: We use `icet` (if available) or `ase.build.sqs` to find the best atomic ordering that matches the random correlation functions.
4.  **Deterministic Randomness**: All generation logic must accept a `seed` or a `numpy.random.Generator` instance. This ensures that if we re-run the pipeline with the same config, we get the exact same set of structures. This is vital for debugging.

### 3.2. Consumers and Producers

-   **Consumer**: `StructureBuilder` consumes `MLIPConfig` (specifically `generator` section) and the `target_system` definition.
-   **Producer**: `StructureBuilder` produces a list of `ASEAtoms` objects. These are passed to the `DatabaseManager` (from Cycle 01) to be stored as `PENDING`.
-   **Downstream**: In Cycle 03, the Surrogate will query these `PENDING` structures.

## 4. Implementation Approach

### Step 1: Transformation Logic (The "Math" Layer)
We start by implementing `transformations.py`. These are pure functions (no side effects).
-   **Task**: Implement `apply_strain`. It should handle both isotropic expansion and shear strains. We need a helper to generate random strain tensors.
-   **Task**: Implement `apply_rattle`. Simple Gaussian noise.

### Step 2: Defect Logic
We implement `defects.py`.
-   **Task**: Implement `VacancyGenerator`. Iterate through unique Wyckoff sites (using `spglib` via ASE) to avoid generating identical vacancies in symmetric crystals.
-   **Task**: Implement `InterstitialGenerator`. This is harder. We might need Voronoi tessellation to find voids, or use `pymatgen.analysis.defects`. For Cycle 02, a random insertion with a minimum distance check is a sufficient MVP.

### Step 3: The Builder Orchestrator
We implement `builder.py`.
-   **Task**: Create `StructureBuilder` class.
-   **Method**: `build_initial_set()`. This orchestrates the flow:
    1.  Create Supercell.
    2.  Apply SQS (if alloy).
    3.  Generate variants:
        -   5 Strain variants.
        -   5 Rattle variants.
        -   1 Vacancy variant.
    4.  Return flattened list of Atoms.

### Step 4: Integration with Config
-   **Task**: Update `MLIPConfig` to include `GeneratorConfig`.
-   **Task**: Ensure defaults are sensible (e.g., don't create 1000 rattle variants by default).

## 5. Test Strategy

### 5.1. Unit Testing Approach (Min 300 words)
We verify the physical correctness of the transformations.

-   **Strain Test**:
    -   Create a unit cube (1x1x1).
    -   Apply 10% tensile strain in X.
    -   Assert new cell vector is [1.1, 0, 0].
    -   Assert volume increased by roughly 10%.
-   **Rattle Test**:
    -   Create a grid of atoms.
    -   Apply rattle with $\sigma=0.1$.
    -   Assert positions have changed.
    -   Assert mean displacement is close to 0 (noise is centred).
-   **Symmetry Test (Vacancy)**:
    -   Create a perfect FCC crystal (all atoms equivalent).
    -   Request vacancies.
    -   Assert only **one** unique vacancy structure is returned (due to symmetry).
-   **Reproducibility Test**:
    -   Run `apply_rattle` with seed=42 twice.
    -   Assert exact bitwise equality of positions.

### 5.2. Integration Testing Approach (Min 300 words)
We verify the pipeline from Config to List[Atoms].

-   **Pipeline Flow**:
    -   Define a `GeneratorConfig` with `num_sqs=2`, `strain_count=3`.
    -   Instantiate `StructureBuilder`.
    -   Call `build()`.
    -   Assert returned list length = $2 \times (1 + 3) = 8$ (Base + Strains) - *Formula depends on exact logic*.
-   **Physical Sanity Check**:
    -   Check that no two atoms in the generated set are closer than 0.5 Angstroms (prevent nuclear fusion).
    -   Check that cell volumes are within physically reasonable bounds (e.g., not compressed to density of black hole).
-   **Metadata Check**:
    -   Verify that each generated atom object has `info['config_type']` set (e.g., 'strain_0.05', 'vacancy_Fe'). This is crucial for tracking data provenance in the database later.
