# Cycle 05 Specification: Active Learning Strategy (Selection & DB)

## 1. Summary

Cycle 05 is the "Brain" upgrade. We move from individual component testing to integrating them into a cohesive Active Learning Loop. This cycle focuses on the "Selection" phase, which is the critical link between the Dynamics Engine (Cycle 04) and the Oracle (Cycle 02).

When a simulation halts, we have a single "dangerous" structure. Simply adding this one structure to the dataset is often insufficient ("One-Shot Learning" is unstable). Instead, we implement a "Local D-Optimality" strategy: we generate a cloud of candidate structures around the halted configuration (e.g., via random displacements or normal mode analysis) and use linear algebra (MaxVol algorithm via `pace_activeset`) to select the most informative subset. This ensures we learn the local potential energy surface efficiently. We also introduce a `DatabaseManager` to track the lineage of these structures.

## 2. System Architecture

**Files to be created/modified in this cycle are marked in Bold.**

```ascii
src/mlip_autopipec/
├── orchestration/
│   ├── **__init__.py**
│   ├── **workflow.py**         # Workflow state machine
│   ├── **selection.py**        # Candidate generation & selection logic
│   └── **database.py**         # Data tracking (SQLite/File-based)
└── app.py                      # (Updated to use WorkflowManager)
```

## 3. Design Architecture

### `WorkflowManager` (in `orchestration/workflow.py`)
*   **Responsibility**: Manage the high-level loop.
*   **States**: `EXPLORATION` -> `SELECTION` -> `CALCULATION` -> `TRAINING` -> `DEPLOYMENT`.
*   **Logic**:
    *   Calls `LammpsRunner`.
    *   If halted, calls `CandidateSelector`.
    *   Calls `QERunner` (Oracle) for the selected candidates.
    *   Calls `PacemakerWrapper` (Trainer).

### `CandidateSelector` (in `orchestration/selection.py`)
*   **Responsibility**: Generate and filter candidates.
*   **Method**: `process_halt(halted_structure) -> List[Atoms]`
*   **Logic**:
    1.  **Generate**: Create $N$ perturbations of the halted structure (e.g., random noise $\pm 0.05$ Å).
    2.  **Select**: Use `pace_activeset` (via `PacemakerWrapper`) to select $k$ best structures from these $N$ candidates + original.
    3.  **Embed**: Pass these $k$ structures to `PeriodicEmbedding` (from Cycle 02) to prepare them for DFT.

### `DatabaseManager` (in `orchestration/database.py`)
*   **Responsibility**: Persist the state of structures.
*   **Schema**:
    *   `structure_id`: Unique ID.
    *   `parent_id`: ID of the halted structure (if applicable).
    *   `origin`: "exploration", "perturbation".
    *   `status`: "pending_dft", "calculated", "trained".
    *   `energy`, `forces`: The DFT results.

## 4. Implementation Approach

1.  **Database**: Implement a lightweight SQLite wrapper or JSON-based tracker in `DatabaseManager`.
2.  **Selection Logic**: Implement `CandidateSelector`. The perturbation logic is simple numpy addition. The integration with `pace_activeset` requires careful file handling.
3.  **Workflow**: Implement the state machine in `WorkflowManager`. This replaces the skeleton `Orchestrator` from Cycle 01 with real logic.

## 5. Test Strategy

### Unit Testing
*   **Selector**:
    *   Give a structure.
    *   Assert `generate_candidates` returns N different structures.
    *   Assert the selection returns a subset.
*   **Database**:
    *   Add a structure. Update its status. Query it.

### Integration Testing
*   **The Loop (Mocked)**:
    *   Simulate `LammpsRunner` returning a halt.
    *   Verify `CandidateSelector` is called.
    *   Verify `QERunner` is called with the output of Selector.
    *   Verify `DatabaseManager` records the new entries.
