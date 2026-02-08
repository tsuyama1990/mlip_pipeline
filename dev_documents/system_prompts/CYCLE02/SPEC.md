# Cycle 02 Specification: Data Management & Structure Generator

## 1. Summary
Cycle 02 focuses on the "Input" side of the active learning loop. We will implement a robust `Dataset` class to manage atomic structures efficiently (using JSONL or Pickle) and the `StructureGenerator` component. The Generator will be upgraded from a mock to a functional component capable of creating structures based on an "Adaptive Exploration Policy" (e.g., varying temperature, pressure, or defects based on configuration).

## 2. System Architecture

### 2.1 File Structure
**Bold** files are to be created or modified in this cycle.

```ascii
.
├── src/
│   └── mlip_autopipec/
│       ├── core/
│       │   ├── **dataset.py**          # Dataset Management Class
│       │   └── orchestrator.py         # Update to use Dataset
│       ├── components/
│       │   ├── generator/
│       │   │   ├── **base.py**         # Base Generator Logic
│       │   │   ├── **md_generator.py** # MD-based Generator (LAMMPS wrapper or ASE)
│       │   │   └── **random_generator.py** # Simple Random/Perturbation Generator
│       ├── domain_models/
│       │   ├── **structure.py**        # Enhance with validation logic
│       │   └── **dataset_config.py**   # Config for Dataset
```

## 3. Design Architecture

### 3.1 Dataset Management (`src/mlip_autopipec/core/dataset.py`)
*   **Requirements**:
    *   Must handle large datasets without loading everything into RAM (streaming).
    *   Must support appending new structures.
    *   Must be able to save/load from disk (recommended format: Extended XYZ or JSONL with `ase` compatibility).
*   **Class `Dataset`**:
    *   `__init__(filepath: Path)`
    *   `append(structures: List[Structure])`
    *   `__iter__() -> Iterator[Structure]`
    *   `__len__() -> int`
    *   `save()` / `load()`

### 3.2 Structure Generator (`src/mlip_autopipec/components/generator/`)
The generator proposes new candidate structures for the active learning loop.

*   **Adaptive Exploration Policy**:
    *   Instead of hardcoded parameters, the generator should accept a policy (e.g., "high temperature scan", "high pressure scan", "defect introduction").
    *   For this cycle, we will implement:
        1.  **Random Perturbation**: Takes initial structures and rattles atoms.
        2.  **Supercell Creation**: Creates supercells with vacancies or substitutions.
        3.  **MD Sampling (Stub)**: Prepare the interface for MD-based sampling (actual LAMMPS execution comes in Cycle 05, but we can use ASE MD here for simple cases).

### 3.3 Domain Models Update
*   **`Structure`**:
    *   Add methods for `to_ase()` and `from_ase()` to interact with the ASE library.
    *   Add validation for `pbc` (must be 3 booleans).

## 4. Implementation Approach

1.  **Dataset Implementation**:
    *   Create `core/dataset.py`.
    *   Implement JSONL storage (each line is a JSON representation of a Structure). This allows appending without rewriting the whole file.
2.  **Structure Model Enhancement**:
    *   Update `domain_models/structure.py` with `ase` conversion methods.
3.  **Generator Implementation**:
    *   Implement `RandomStructureGenerator` which takes a seed structure and applies Gaussian noise to positions.
    *   Implement `SupercellGenerator` which creates $N \times N \times N$ supercells.
4.  **Orchestrator Integration**:
    *   Update `Orchestrator` to instantiate the chosen Generator based on config.
    *   Update the loop to store generated structures into the `Dataset`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Dataset**:
    *   Test appending and reading back structures.
    *   Test persistence (saving to file and reloading).
    *   Test handling of large lists (mocking memory constraints).
*   **Structure**:
    *   Test `to_ase` and `from_ase` round-trip conversion.
*   **Generator**:
    *   Test `RandomStructureGenerator` produces structures with different positions but same composition.

### 5.2 Integration Testing
*   **Generator -> Dataset**: Run a flow where the Generator creates 100 structures, and they are successfully saved to a Dataset file on disk.
