# Cycle 02 Specification: Data Management & Structure Generation

## 1. Summary
This cycle implements the core data structures (`Structure`, `Dataset`) and the `StructureGenerator`. The focus is on robustly handling atomic configurations, saving/loading them to disk (using ASE's extended XYZ format or JSONL), and generating initial random structures to seed the active learning loop.

## 2. System Architecture

### 2.1. File Structure
The following file structure must be created/modified. **Bold** files are to be implemented in this cycle.

```ascii
src/mlip_autopipec/
├── domain_models/
│   ├── structure.py        # Enhanced with validation
│   └── **dataset.py**      # Dataset Management
├── components/
│   ├── generator/
│   │   ├── **__init__.py**
│   │   ├── **base.py**     # Base Generator
│   │   ├── **random.py**   # Random Structure Generator
│   │   └── **rattle.py**   # Rattle/Displace Generator
│   └── oracle/
│       └── ...
tests/
    ├── **test_generator.py**
    └── **test_dataset.py**
```

### 2.2. Class Blueprints

#### `src/mlip_autopipec/domain_models/dataset.py`
```python
from pathlib import Path
from typing import Iterator
from mlip_autopipec.domain_models.structure import Structure

class Dataset:
    def __init__(self, filepath: Path):
        self.filepath = filepath

    def append(self, structure: Structure) -> None:
        """Append structure to file (JSONL or extxyz)."""
        pass

    def __iter__(self) -> Iterator[Structure]:
        """Stream structures from file."""
        pass

    def __len__(self) -> int:
        """Count structures (efficiently)."""
        pass
```

## 3. Design Architecture

### 3.1. Domain Models
*   **`Structure`**: Enhance `src/mlip_autopipec/domain_models/structure.py` to support conversion to/from `ase.Atoms`. This is crucial for interoperability with ASE-based tools (Quantum Espresso, LAMMPS).
*   **`Dataset`**: Abstraction over a file on disk. We avoid loading all structures into memory.
    *   **Format**: JSON Lines (`.jsonl`) is preferred for metadata flexibility, or Extended XYZ (`.extxyz`) for compatibility. JSONL with Pydantic serialization is more robust for our internal pipeline.

### 3.2. Generator Components (`src/mlip_autopipec/components/generator/`)
*   **`RandomGenerator`**:
    *   Input: Composition (e.g., `{"Fe": 1, "Pt": 1}`), Supercell size, Count.
    *   Logic: Place atoms randomly in a box.
    *   **Constraint**: Check minimum interatomic distance (`min_dist`). If atoms are too close, retry placement (Simple Random Sequential Adsorption).
*   **`RattleGenerator`**:
    *   Input: Seed structure, standard deviation `sigma`.
    *   Logic: Apply Gaussian noise to positions and cell vectors.

## 4. Implementation Approach

1.  **Structure Model**: Add `from_ase(atoms)` and `to_ase()` methods to `Structure`.
2.  **Dataset**: Implement `Dataset` class using `jsonl`. Use `pydantic.TypeAdapter` for serialization.
3.  **Generators**:
    *   Implement `RandomGenerator` with collision detection.
    *   Implement `RattleGenerator` for augmenting existing structures.
4.  **CLI Integration**: Add a `generate` subcommand to the CLI for debugging: `mlip-pipeline generate --config config.yaml`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_dataset.py`**:
    *   Create a temporary file.
    *   Append 100 structures.
    *   Iterate and verify data integrity.
*   **`test_generator.py`**:
    *   `test_random_generation`: Generate 10 structures. Check that no two atoms are closer than `min_dist` (e.g., 1.5 Å).
    *   `test_rattle`: Verify displaced positions are close to original but not identical.

### 5.2. Integration Testing
*   **`test_orchestrator_integration`**:
    *   Configure Orchestrator with `RandomGenerator`.
    *   Run one cycle.
    *   Verify `Dataset` file is created and populated with generated structures.
