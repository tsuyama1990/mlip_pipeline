# Cycle 02 Specification: Data Management & Structure Generation

## 1. Summary
This cycle implements real structure generation and data management logic. We replace the mock structure generator with physical ones (`RandomStructureGenerator`, `HeuristicGenerator`) and implement a persistent file-based database system for `Dataset`. We also implement the `Selector` component to filter candidate structures using basic strategies (Random/Greedy) before passing them to the expensive Oracle.

## 2. System Architecture

### 2.1. File Structure
The following files must be created or modified. **Bold** files are the focus of this cycle.

src/mlip_autopipec/
├── domain_models/
│   ├── **structure.py**            # Enhanced Structure & Dataset Models
├── interfaces/
│   ├── **generator.py**            # Enhanced BaseStructureGenerator
│   ├── **selector.py**             # BaseSelector
├── infrastructure/
│   ├── **generator/**
│   │   ├── **__init__.py**
│   │   ├── **random_generator.py** # Random Structure Generator
│   │   └── **heuristic_generator.py** # Heuristic Structure Generator
│   ├── **selector/**
│   │   ├── **__init__.py**
│   │   └── **simple_selector.py**  # Random/Greedy Selector
│   ├── **database/**
│   │   ├── **__init__.py**
│   │   └── **file_database.py**    # File-based Dataset Management
└── orchestrator/
    └── **simple_orchestrator.py**  # Update logic to use Generator/Selector

### 2.2. Class Diagram
*   `RandomStructureGenerator` implements `BaseStructureGenerator`.
*   `SimpleSelector` implements `BaseSelector`.
*   `FileDatabase` manages `datasets/` directory and `.pckl.gzip` files.

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/structure.py`)
*   **`Structure`**: Enhance with `properties` dict for metadata (e.g., "generation_method": "random").
*   **`Dataset`**: Manage a collection of `Structure` objects.
    *   Methods: `save(path)`, `load(path)`, `append(structures)`, `merge(other)`.
    *   Persistence: Use `pickle` + `gzip` for efficient storage of ASE objects.

### 3.2. Structure Generation (`infrastructure/generator/`)
*   **`RandomStructureGenerator`**:
    *   Input: Composition (e.g., {"Fe": 1, "Pt": 1}), Volume scaling.
    *   Logic: Generate random unit cells within constraints. Check min/max distance.
    *   Use `ase.build` utilities.
*   **`HeuristicGenerator`**:
    *   Logic: Rattle existing structures, generate supercells, or apply strain.

### 3.3. Selector (`infrastructure/selector/`)
*   **`SimpleSelector`**:
    *   Logic: Select `N` structures from `M` candidates based on strategy (Random, Greedy Farthest Point using simple features).
    *   Goal: Reduce the number of expensive DFT calls.

## 4. Implementation Approach

1.  **Enhance Structure Model**: Update `domain_models/structure.py` to support serialization/deserialization.
2.  **Implement File Database**: Create `infrastructure/database/file_database.py` to handle loading/saving datasets.
3.  **Implement Generators**: Create `infrastructure/generator/*.py`. Use `ase.build` and `numpy.random`.
4.  **Implement Selector**: Create `infrastructure/selector/simple_selector.py`.
5.  **Update Factory**: Register new components in `factory.py`.
6.  **Update Orchestrator**: Integrate Generator and Selector into the loop.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Generator Test**: Verify `RandomStructureGenerator` produces valid ASE atoms with correct composition and non-overlapping atoms.
*   **Database Test**: Verify `Dataset.save()` and `Dataset.load()` round-trip correctly.
*   **Selector Test**: Verify `SimpleSelector` picks the requested number of structures.

### 5.2. Integration Testing
*   **Gen -> Select -> Save Loop**:
    *   Generate 100 structures.
    *   Select 10.
    *   Save to `test_dataset.pckl.gzip`.
    *   Load and verify count is 10.
