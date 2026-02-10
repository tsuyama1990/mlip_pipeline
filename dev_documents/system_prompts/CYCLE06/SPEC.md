# Cycle 06: The OTF Loop Integration

## 1. Summary

Cycle 06 is the "Integration Cycle" where we connect all the components developed in previous cycles (Generator, Oracle, Trainer, Dynamics) into a coherent, self-driving Active Learning loop.

The core logic is the "Halt & Diagnose" workflow:
1.  **Exploration**: The Dynamics Engine runs MD.
2.  **Detection**: If high uncertainty is detected (Halt), the loop pauses.
3.  **Selection**: The `StructureGenerator` creates *local candidates* around the high-uncertainty region (e.g., using normal mode displacement).
4.  **Calculation**: The `Oracle` computes DFT energies for these candidates (after embedding).
5.  **Refinement**: The `Trainer` updates the potential.
6.  **Resume**: The simulation resumes from the halt point with the improved potential.

This cycle transforms the static "Mock Loop" from Cycle 01 into the real, dynamic system.

## 2. System Architecture

The file structure remains largely the same, but the `Orchestrator` logic becomes significantly more complex. We may introduce a new `OTFLoop` helper class to manage the state of a single active learning iteration.

```
.
├── config.yaml
├── src/
│   └── mlip_autopipec/
│       ├── core/
│       │   ├── **orchestrator.py** # Major Update
│       │   └── **otf_loop.py**     # New Loop Logic
│       ├── components/
│       │   ├── **generator.py**    # Update: Local candidate generation
│       └── utils/
│           └── **file_manager.py** # Helper for directory structure
└── tests/
    ├── **test_otf_loop.py**
    └── **test_orchestrator_integration.py**
```

## 3. Design Architecture

### OTF Loop Logic (`core/otf_loop.py`)
This class encapsulates the state of one "Active Learning Iteration" (e.g., `iter_005`).
*   `run_iteration(iteration_index, current_potential)`:
    1.  Create iteration directory.
    2.  Run MD (`dynamics.explore`).
    3.  If `halted`:
        *   Load halt snapshot.
        *   `generator.generate_local_candidates(snapshot)`.
        *   `oracle.compute(candidates)`.
        *   `trainer.update_dataset(results)`.
        *   `trainer.train(initial_potential=current_potential)`.
        *   Return `new_potential`.
    4.  Else:
        *   Return `current_potential` (Converged).

### Local Candidate Generation (`components/generator.py`)
Add method `generate_local_candidates(structure, n_candidates)`:
*   Identify the atom with max $\gamma$.
*   Apply perturbations (random displacement, normal modes) *localized* around that atom.
*   Return a list of structures.

### Resume Logic
The `DynamicsEngine` needs to support `restart` files.
*   `explore(potential, restart_file=None)`:
    *   If `restart_file` provided, use `read_restart` in LAMMPS.
    *   Else, use `read_data`.

## 4. Implementation Approach

1.  **OTF Loop Class**: Implement `core/otf_loop.py`. Move the heavy lifting out of `Orchestrator.run()`.
2.  **Local Generation**: Implement `generate_local_candidates` in `StructureGenerator`.
3.  **Resume Support**: Update `LAMMPSDynamics` to handle restarts.
4.  **Orchestrator Update**: Refactor `Orchestrator` to loop over `OTFLoop.run_iteration()` until convergence or max cycles.
5.  **Logging**: Enhance logging to track the "Halt -> Retrain -> Resume" flow clearly.

## 5. Test Strategy

### Unit Testing
*   **OTF Logic**: Test `OTFLoop.run_iteration` with Mock components.
    *   Mock `dynamics.explore` to return `halted=True` on the first call, `halted=False` on the second.
    *   Assert that `generator`, `oracle`, and `trainer` are called in the correct order.
    *   Assert that the potential is updated.

### Integration Testing (System Level)
*   **Full Cycle Test (Mock)**:
    *   Run the full `main.py` with `config_mock.yaml`.
    *   Configure the Mock Dynamics to halt 3 times.
    *   Verify that 3 retraining cycles occur and 3 new potential files are generated.
    *   Verify the final state is "Converged".

### Integration Testing (Real - Optional)
*   **Toy System**: Run Argon MD with a very poor initial potential (e.g., trained on 1 atom).
*   **Observation**: Watch it halt immediately, retrain on the collision, and then run longer.
