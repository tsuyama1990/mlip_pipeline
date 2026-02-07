# Cycle 02 Specification: Core Orchestration & Data Management

## 1. Summary

**Goal**: Implement the "Brain" of the system (`SimpleOrchestrator`) and establish a robust data management layer. This cycle transforms the loose collection of mock components into a coherent, automated pipeline. The Orchestrator will manage the Active Learning loop, handle state transitions, and ensure data integrity (saving/loading datasets) without crashing. Additionally, the `StructureGenerator` will be fleshed out with basic policies (Random Displacement) to support the loop.

**Key Deliverables**:
1.  **`SimpleOrchestrator`**: The main loop logic (Explore -> Detect -> Select -> Label -> Train).
2.  **`Dataset` Management**: Utilities to save/load lists of `Structure` objects to disk (JSON/Pickle/ExtXYZ) efficiently.
3.  **`StructureGenerator`**: Implementation of `RandomDisplacement` and `NormalMode` generation strategies.
4.  **State Persistence**: Ability to resume a run from the last checkpoint.

## 2. System Architecture

Files in **bold** are the primary focus of this cycle.

```ascii
src/mlip_autopipec/
├── domain_models/
│   ├── **dataset.py**        # Dataset class (List[Structure] wrapper)
│   └── ...
├── orchestrator/
│   ├── **__init__.py**
│   └── **simple_orchestrator.py** # The main loop logic
├── infrastructure/
│   ├── generator/
│   │   ├── **__init__.py**
│   │   └── **random.py**     # RandomDisplacement generator
│   └── ...
└── utils/
    └── **io.py**             # File I/O utilities (ASE conversion)
```

## 3. Design Architecture

### 3.1 `SimpleOrchestrator`

The `SimpleOrchestrator` class manages the lifecycle of a project.

*   **State**: `iteration` (int), `current_potential` (Potential), `dataset` (Dataset).
*   **Methods**:
    *   `run()`: The main entry point. Loops until `max_cycles` or convergence.
    *   `_step_exploration()`: Calls Dynamics Engine. Handles halt logic.
    *   `_step_generation()`: Calls Structure Generator.
    *   `_step_labelling()`: Calls Oracle.
    *   `_step_training()`: Calls Trainer.
*   **Error Handling**: Wraps each step in try-except blocks to log errors without crashing the entire pipeline (unless critical).

### 3.2 Data Management (`Dataset`)

*   **Requirement**: Efficiently handle thousands of structures.
*   **Implementation**: A wrapper around `List[Structure]`.
    *   `load(path: Path)`: Loads from JSONL or Pickle.
    *   `save(path: Path)`: Saves to disk.
    *   `merge(other: Dataset)`: Combines two datasets, removing duplicates (based on hash).
    *   `to_ase()`: Converts to list of `ase.Atoms` for compatibility.

### 3.3 Structure Generator Policies

*   **`RandomDisplacement`**:
    *   Input: A seed structure.
    *   Logic: Displace each atom by a random vector sampled from a Gaussian distribution ($\sigma = 0.01 \sim 0.1$ Å).
    *   Constraint: Ensure minimum distance is not violated (simple check).

## 4. Implementation Approach

1.  **Implement `Dataset`**: Create the data container and I/O utilities in `domain_models` and `utils`.
2.  **Implement `StructureGenerator`**: Create `infrastructure/generator/random.py`.
3.  **Implement `SimpleOrchestrator`**:
    *   Initialise components using a factory pattern based on Config.
    *   Implement the loop logic.
    *   Add logging for state transitions ("Starting Iteration 1...", "Halted at step 500").
4.  **Update `main.py`**: Connect the CLI `run` command to `SimpleOrchestrator.run()`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Orchestrator Logic**: Mock all components. Verify that `run()` calls `dynamics.run()`, then `generator.generate()`, etc., in the correct order.
*   **Data Persistence**: Save a `Dataset` to a temp file, load it back, and assert equality.
*   **Generator**: Verify `RandomDisplacement` actually moves atoms and returns distinct structures.

### 5.2 Integration Testing
*   **"The Mock Loop" (Automated)**:
    *   Configure `SimpleOrchestrator` with `Mock` components.
    *   Run for 2 cycles.
    *   Assert that `dataset.json` grows in sise.
    *   Assert that `iteration` counter increments.
    *   Assert that `current_potential` updates.
