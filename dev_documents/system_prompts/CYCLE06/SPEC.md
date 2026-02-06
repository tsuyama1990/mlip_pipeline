# Cycle 06 Specification: Active Learning Loop (The Halt Mechanism)

## 1. Summary

Cycle 06 is where the system becomes "Autonomous." We implement the **Active Learning Loop**. The key mechanism is the **Uncertainty Watchdog**.

During MD, the system monitors the "Extrapolation Grade" ($\gamma$). If the simulation enters a region where the potential is unreliable ($\gamma > \gamma_{thresh}$), the simulation is **Halted**. The Orchestrator then intervenes, extracts the problematic structure, labels it (Oracle), retrains the potential (Trainer), and resumes the simulation. This cycle implements the "Halt" detection and the Orchestrator's response logic.

## 2. System Architecture

Files to create/modify are **bolded**.

```ascii
src/mlip_autopipec/
├── config/
│   └── config_model.py         # Update ExplorerConfig (uncertainty_threshold)
└── infrastructure/
    └── dynamics/
        └── **lammps_adapter.py** # Update to support fix halt
```

## 3. Design Architecture

### 3.1. Watchdog Configuration
*   `uncertainty_threshold`: float (e.g., 5.0).
*   `check_interval`: int (e.g., every 10 steps).

### 3.2. Orchestrator Logic Update
*   **State Machine**:
    *   `EXPLORE`: Run MD.
    *   If `result.halted`: Transition to `SELECT`.
    *   `SELECT`: Extract structure -> Generate candidates (Rattling). Transition to `LABEL`.
    *   `LABEL`: Oracle computes. Transition to `TRAIN`.
    *   `TRAIN`: Update potential. Transition to `EXPLORE` (Resume).

### 3.3. LAMMPS Implementation
*   Use `compute pace` (part of USER-PACE package).
*   Use `fix halt` condition: `v_max_gamma > ${threshold}`.

## 4. Implementation Approach

1.  **LAMMPS Command**: Add `compute pace ... gamma_mode=1` and `fix halt` to the input template.
2.  **Return Code Handling**: If LAMMPS exits with the specific "Halt" error code, `LammpsAdapter` must return `ExplorationResult(halted=True)`.
3.  **Extraction**: Implement logic to read the *last* frame of the dump file (the one that triggered the halt).
4.  **Orchestrator Wiring**: Connect the pieces.
    *   `if result.halted:`
    *   `candidates = generator.generate_local_candidates(result.structure)`
    *   `labeled = oracle.compute(candidates)`
    *   `trainer.train(labeled)`

## 5. Test Strategy

### 5.1. Unit Testing
*   **Input**: Verify `fix halt` command exists in `in.lammps`.

### 5.2. Integration Testing (Mocked Halt)
*   **Mock**: Configure `MockExplorer` to simulate a Halt event after 50 steps.
*   **Run**: Execute Orchestrator.
*   **Assert**:
    1.  The loop detects the halt.
    2.  It calls `Oracle.compute` with the halted structure.
    3.  It calls `Trainer.train`.
    4.  It calls `Explorer.explore` again (Resume).
