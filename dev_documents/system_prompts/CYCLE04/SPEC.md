# Cycle 04: Active Learning Loop (Integration)

## 1. Summary

Cycle 04 is the integration phase where the individual components built in the previous cycles (Oracle, Database, Trainer, Dynamics) are connected to form the autonomous "Active Learning Loop". The primary goal is to implement the logic that drives the self-improvement of the potential.

This includes:
1.  **Orchestrator Logic**: The state machine that transitions between Exploration, Selection, Calculation, and Refinement.
2.  **Selection Strategy**: The "Periodic Embedding" technique to extract local, uncertain structures from halted MD simulations and prepare them for DFT.
3.  **Active Set Optimization**: Implementing D-Optimality to select only the most informative structures, preventing database bloating.

By the end of this cycle, the system will be able to perform a complete closed-loop iteration: Run MD -> Detect Halt -> Extract Structure -> Run DFT -> Retrain -> Resume MD.

## 2. System Architecture

### File Structure

**mlip_autopipec/**
├── **orchestrator/**
│   ├── **loop.py**             # Main Active Learning Loop
│   ├── **state.py**            # WorkflowState management
│   └── **selection.py**        # Structure extraction & embedding
└── **trainer/**
    └── **active_set.py**       # Active Set Selection wrapper

### Component Description

*   **`orchestrator/loop.py`**: The "Conductor". It runs a `while` loop that continues until the potential converges or the max iterations are reached. It calls the sub-modules in sequence.
*   **`orchestrator/selection.py`**: Handles the "Surgery" on atoms. When MD halts, it takes the full simulation box, identifies the high-$\gamma$ atom, cuts out a cluster, and wraps it in a new periodic box (Periodic Embedding) suitable for DFT.
*   **`trainer/active_set.py`**: Wraps the `pace_activeset` command. It takes a pool of candidate structures and selects a subset that maximizes the information gain (D-optimality).

## 3. Design Architecture

### Domain Models

**`WorkflowState`**
*   **Role**: Persists the progress of the workflow.
*   **Fields**:
    *   `cycle_index`: `int`
    *   `current_phase`: `Enum` (Exploration, Labeling, Training)
    *   `latest_potential_path`: `FilePath`
    *   `converged`: `bool`

**`SelectionConfig`**
*   **Role**: Parameters for structure extraction.
*   **Fields**:
    *   `cutoff_radius`: `float` (e.g., 5.0 Å)
    *   `buffer_thickness`: `float` (e.g., 2.0 Å)
    *   `max_structures_per_halt`: `int`

### Key Invariants
1.  **Continuity**: The loop must be resumable. If the process is killed, restarting it should load the `WorkflowState` and continue from the last successful step.
2.  **Periodicity**: Extracted structures must always be periodic (using the embedding box), otherwise DFT plane-wave codes like Quantum Espresso cannot calculate them accurately.
3.  **Monotonicity**: The database size should monotonically increase (or stay same), never decrease.

## 4. Implementation Approach

1.  **Periodic Embedding Logic**:
    *   Implement `embed_local_region(full_atoms, center_atom_index, radius)`.
    *   Use `ase` to cut the cluster.
    *   Define a new bounding box that creates sufficient vacuum (or buffer) to avoid spurious interactions, but mark it as periodic for QE.

2.  **Active Set Wrapper**:
    *   Implement `select_active_set(candidates, current_potential)`.
    *   Run `pace_activeset` to calculate the MaxVol selection.

3.  **Orchestrator Main Loop**:
    *   Implement the finite state machine:
        ```python
        while not state.converged:
            if state.phase == EXPLORATION:
                status = lammps_runner.run(...)
                if status.halted:
                    candidates = selection.extract(status)
                    state.phase = SELECTION
            elif state.phase == SELECTION:
                 selected = active_set.select(candidates)
                 state.phase = CALCULATION
            elif state.phase == CALCULATION:
                 results = oracle.compute(selected)
                 database.add(results)
                 state.phase = TRAINING
            elif state.phase == TRAINING:
                 pot = trainer.train(database)
                 state.phase = EXPLORATION
        ```

## 5. Test Strategy

### Unit Testing
*   **Selection**: Create a dummy atoms object (e.g., 1000 atoms). Select an atom in the center. Verify `embed_local_region` returns a smaller atoms object (e.g., ~50 atoms) that contains the center atom and its neighbors.
*   **State Persistence**: Create a `WorkflowState`, save it to JSON, load it back, and assert equality.

### Integration Testing
*   **Mocked Loop**: Run the Orchestrator with mocked sub-components.
    *   `LammpsRunner` returns `halted=True` on the 1st call, `halted=False` on the 2nd.
    *   `Oracle` returns dummy energies immediately.
    *   Verify that the state transitions correctly from Exploration -> ... -> Training -> Exploration -> Completed.
