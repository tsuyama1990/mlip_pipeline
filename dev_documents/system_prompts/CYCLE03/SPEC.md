# Specification: Cycle 03 - The Explorer (Structure Generation)

## 1. Summary

Cycle 03 introduces the "Explorer" capability to the system. While the previous cycles established the infrastructure and the "Oracle" (DFT), the system currently lacks the ability to *ask questions*. It relies on pre-existing datasets. In this cycle, we implement the mechanisms to actively generate new atomic structures that explore the potential energy surface.

This is a critical pivot point where the system transitions from "Passive Training" to "Active Learning". The core innovation here is the **Adaptive Exploration Policy**. Instead of hardcoding "run MD at 300K", the system will analyze the input material (e.g., estimating melting point or band gap) and decide on a strategy. For example, if the material is an insulator, it might favor creating charged defects. If it is a hard ceramic, it might prioritize high-strain configurations.

The output of this cycle is a module that accepts a "Seed Structure" and returns a list of "Candidate Structures" ready for DFT calculation, effectively closing the loop for the next cycle.

## 2. System Architecture

### 2.1 File Structure

Files to be created or modified (bold):

```
mlip-autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── config/
│       │   └── **config_model.py**         # Update with StructureGenConfig
│       ├── domain_models/
│       │   └── **exploration.py**          # ExplorationTask definitions
│       ├── physics/
│       │   └── structure_gen/
│       │       ├── __init__.py
│       │       ├── **generator.py**        # The Factory and Base Classes
│       │       ├── **strategies.py**       # Specific strategies (Strain, Defect)
│       │       ├── **policy.py**           # The Adaptive Policy Logic
│       │       └── **embedding.py**        # Periodic Embedding Logic
│       └── utils/
│           └── **crystallography.py**      # Helpers for symmetry/supercells
├── tests/
│   ├── unit/
│   │   ├── **test_policy.py**
│   │   └── **test_embedding.py**
│   └── integration/
│       └── **test_generation_pipeline.py**
└── config.yaml
```

### 2.2 Component Blueprints

#### `src/mlip_autopipec/domain_models/exploration.py`

```python
from enum import Enum
from pydantic import BaseModel

class ExplorationMethod(str, Enum):
    MD = "molecular_dynamics"
    STATIC = "static_displacement"
    AKMC = "adaptive_kmc"

class ExplorationTask(BaseModel):
    method: ExplorationMethod
    parameters: dict  # e.g., {'temperature': 1000, 'pressure': 0}
    modifiers: list[str] = [] # e.g., ['defect:vacancy', 'strain:shear']
```

#### `src/mlip_autopipec/physics/structure_gen/policy.py`

```python
class AdaptivePolicy:
    def decide_strategy(self, structure: Atoms, current_uncertainty: float) -> List[ExplorationTask]:
        """
        Analyzes the state and returns a list of tasks.
        Example: If uncertainty is high in high-T regions, propose MD at T=0.8*Tm.
        """
        tasks = []
        if self._is_metal(structure):
             tasks.append(ExplorationTask(method=ExplorationMethod.MD, ...))
        else:
             tasks.append(ExplorationTask(method=ExplorationMethod.STATIC, modifiers=['defect']))
        return tasks
```

#### `src/mlip_autopipec/physics/structure_gen/embedding.py`

```python
def extract_periodic_box(large_atoms: Atoms, center_index: int, cutoff: float) -> Atoms:
    """
    Cuts a cluster around 'center_index' and wraps it into a
    minimal periodic supercell that respects the cutoff radius.
    """
    # ... Implementation of the "Rectangular Box" extraction logic ...
```

## 3. Design Architecture

### 3.1 The Strategy Pattern
We use the **Strategy Pattern** for structure generation.
*   **`StructureGenerator` Protocol**: Defines a `generate(atoms, count) -> List[Atoms]` interface.
*   **Concrete Strategies**:
    *   `RandomSliceGenerator`: Takes a large MD dump and randomly picks frames.
    *   `StrainGenerator`: Applies affine deformations (Voigt strain) to the unit cell.
    *   `DefectGenerator`: Removes atoms (vacancies) or swaps them (antisites) based on Wyckoff positions.

### 3.2 Periodic Embedding (The "Cookie Cutter")
One of the hardest problems in Active Learning is taking a localized failure (e.g., a dislocation core in a 10,000 atom simulation) and turning it into a tractable DFT calculation (100 atoms).
*   We cannot simply cut a sphere (cluster) because standard plane-wave DFT (Quantum Espresso) requires periodic boundary conditions.
*   **Design**: The `EmbeddingHandler` identifies the "bad" atom, defines an orthorhombic box around it (size > 2 * Cutoff), and populates this box with the atoms from the large simulation. This ensures that the local environment seen by the central atom is identical to the large simulation, but the total atom count is small enough for DFT.

## 4. Implementation Approach

### Step 1: Embedding Logic
1.  Implement `extract_periodic_box` in `embedding.py`. This is purely geometric logic using ASE.
2.  **Verify**: Create a test with a known lattice, cut a box, and verify the distances between atoms across the new periodic boundaries are correct.

### Step 2: Generators
1.  Implement `StrainGenerator`: Use `ase.constraints.StrainFilter` or direct cell manipulation.
2.  Implement `DefectGenerator`: Use `pymatgen` or custom logic to identify unique sites and remove them.

### Step 3: Adaptive Policy
1.  Implement a simple rule-based policy first.
    *   Rule 1: If it's the first cycle, do "Cold Start" (Random distortions).
    *   Rule 2: If previous validation failed on Elasticity, do "Strain Generation".
2.  Integrate `ExplorationTask` into the `Orchestrator`.

### Step 4: Orchestrator Integration
1.  Modify `Orchestrator` to call `policy.decide_strategy()`.
2.  Execute the tasks. For now (Cycle 03), "MD" tasks might just look at existing trajectories or run a dummy function, since the full MD engine comes in Cycle 04. Focus on the *Static* generators (Strain/Defects) which can be fully implemented now.

## 5. Test Strategy

### 5.1 Unit Testing Approach (Min 300 words)
*   **Embedding Geometry**: This is the most critical unit test. We will construct a simple Cubic lattice. We will request an embedded box centered on atom 0. We will verify that the resulting `Atoms` object has a `cell` that is orthogonal and sufficiently large. We will check that the number of atoms in the cut box matches the theoretical density.
*   **Policy Logic**: We will feed dummy structures (one labeled "Metal", one "Insulator") into the `AdaptivePolicy` and assert that the returned `ExplorationTask` list contains the expected methods (e.g., Metal -> MD, Insulator -> Defect).
*   **Strain correctness**: We will apply a 1% uniaxial strain using `StrainGenerator`. We will assert that the lattice vector `a` has increased by exactly 1% and `b`, `c` are unchanged.

### 5.2 Integration Testing Approach (Min 300 words)
*   **Generator Pipeline**: We will set up a test pipeline that takes a `cif` file (e.g., TiO2), passes it to the `Orchestrator`, which calls the `Policy`. The Policy selects `DefectGenerator`. The Generator produces 5 labeled structures (Vacancy_Ti, Vacancy_O, etc.).
*   **Success Criteria**:
    1.  The output is a list of valid `ase.Atoms` objects.
    2.  The structures are distinct (check fingerprints or simple composition).
    3.  The structures are "computable" (no overlapping atoms, sensible cell dimensions).
*   **Visual Check**: During development, we will write the generated structures to `debug_candidates.xyz` and manually inspect them with visualization software (OVITO/Vesta) to ensure the embedding didn't create "Frankenstein" structures with broken bonds at the boundaries.
