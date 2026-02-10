# Cycle 06 Specification: On-the-Fly (OTF) Loop Integration

## 1. Summary

This cycle integrates all previous components into a cohesive Active Learning Loop. The Orchestrator is updated to handle the "Halt & Diagnose" workflow: when the Dynamics Engine halts due to high uncertainty, the system must pause, generate local candidates around the problematic structure, label them with the Oracle, retrain the potential, and then resume the simulation.

This is the core "intelligence" of PYACEMAKER, allowing it to autonomously expand its domain of applicability.

## 2. System Architecture

The following file structure will be created. **Bold** files are the targets for this cycle.

```ascii
src/mlip_autopipec/
├── core/
│   └── **orchestrator.py**         # (Major Update: Implement Halt Handling)
├── components/
│   ├── generators/
│   │   └── **adaptive.py**         # (Update: Add Local Candidate Generation)
tests/
└── **test_otf_loop.py**            # Tests for the full feedback loop
```

## 3. Design Architecture

### 3.1. Orchestrator Logic (`core/orchestrator.py`)
*   **`run_cycle()`**: Modified to include the OTF branch.
    *   **Step 1: Explore (Dynamics)**
        *   Call `DynamicsEngine.run(potential)`.
        *   If `halted=False`: Proceed to next iteration (or finish).
        *   If `halted=True`: Enter **Refinement Loop**.
    *   **Step 2: Diagnose & Generate (Refinement)**
        *   Extract `halt_structure`.
        *   Call `StructureGenerator.enhance(halt_structure)` to generate `local_candidates` (e.g., normal mode displacements).
    *   **Step 3: Label (Oracle)**
        *   Call `Oracle.compute(local_candidates)`.
    *   **Step 4: Train (Trainer)**
        *   Update dataset.
        *   Call `Trainer.train(initial_potential=current_potential)` (Fine-tuning).
    *   **Step 5: Resume**
        *   Deploy new potential.
        *   Resume dynamics from the halt point (or restart).

### 3.2. Local Candidate Generation (`components/generators/adaptive.py`)
*   **`AdaptiveGenerator.enhance(structure) -> List[Structure]`**:
    *   Takes a single structure (the one that caused the halt).
    *   Generates 10-20 variations to explore the local energy landscape.
        *   **Random Displacement**: `rattle(stdev=0.05)`
        *   **Volume Strain**: `strain(+- 2%)`
    *   Returns the list of candidates.

## 4. Implementation Approach

1.  **Orchestrator Update**: Rewrite `run_loop` to handle the `DynamicsResult` object.
    *   Implement the branching logic (Converged vs Halted).
    *   Implement the sub-loop for refinement.
2.  **Generator Update**: Implement `enhance` method in `AdaptiveGenerator`.
    *   Simple implementation first: just random rattling.
3.  **State Management**: Ensure `workflow_state.json` tracks "Halt Count" and "Refinement Iterations".

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_generator.py`**:
    *   Test `enhance(structure)`: Returns > 1 structure, all distinct from input.

### 5.2. Integration Testing (Mocked Loop)
*   **`test_otf_loop.py`**:
    *   **Scenario**:
        1.  **Mock Dynamics**: Returns `halted=True` on first call, `halted=False` on second call.
        2.  **Mock Oracle**: Returns energies for candidates.
        3.  **Mock Trainer**: Returns a "new" potential path.
    *   **Verification**:
        *   Assert Orchestrator calls `enhance` -> `Oracle` -> `Trainer` -> `Dynamics(resume)`.
        *   Assert loop completes successfully after one refinement step.
