# CYCLE 04 Specification: Dynamics Engine & Active Learning Loop

## 1. Summary

In this cycle, we implement the **`LammpsDynamics`** engine and close the "Active Learning Loop". The Dynamics Engine runs Molecular Dynamics (MD) simulations using the ACE potential. Crucially, it employs an **Uncertainty Watchdog** that monitors the extrapolation grade ($\gamma$) in real-time. If $\gamma$ exceeds a threshold, the simulation halts, and the problematic structure is sent to the Oracle for labeling.

## 2. System Architecture

Files to be modified/created:

```ascii
src/mlip_autopipec/
├── config/
│   └── **exploration_config.py**   # MD/Watchdog settings
├── dynamics/
│   ├── **__init__.py**
│   └── **lammps_driver.py**        # LAMMPS Wrapper
└── orchestration/
    └── orchestrator.py             # Update loop to handle "Halt" events
```

## 3. Design Architecture

### 3.1. `ExplorationConfig` (Pydantic)
*   Parameters:
    *   `temperature`: float
    *   `pressure`: float
    *   `timestep`: float
    *   `steps`: int
    *   `uncertainty_threshold`: float (default 5.0)

### 3.2. `LammpsDynamics` Class
*   **Interface**: Implements `Explorer` protocol (partially, or as a separate `Dynamics` protocol).
*   **Method**: `run_md(structure, potential_path) -> (FinalStructure, HaltInfo)`
*   **Watchdog Implementation**:
    *   Use `lammps` Python interface or generated input scripts.
    *   Command: `compute g all pace ... gamma_mode=1`
    *   Command: `fix halt all halt 10 v_max_gamma > ${threshold} error hard`
*   **Halt Handling**:
    *   If LAMMPS exits with the specific error code, capture the dump file.
    *   Extract the snapshot corresponding to the high uncertainty moment.

### 3.3. Active Learning Logic (Orchestrator)
*   The `Orchestrator` loop changes from "Generate -> Calc -> Train" to "Explore (MD) -> Check Halt -> (If Halt) Calc -> Train".
*   If no halt occurs (convergence), the cycle finishes or moves to a higher temperature.

## 4. Implementation Approach

1.  **LAMMPS Setup**: Ensure `lammps` with `USER-PACE` package is available (or mocked).
2.  **Driver Logic**:
    *   Create `src/mlip_autopipec/dynamics/lammps_driver.py`.
    *   Write `in.lammps` generation logic, ensuring `pair_style hybrid/overlay` is used (as per Cycle 02).
3.  **Halt & Extract**:
    *   Implement logic to parse LAMMPS log/dump to find the "bad" structure.
4.  **Orchestrator Update**:
    *   Integrate `LammpsDynamics`.
    *   Implement the conditional logic: `if halt_info.halted: oracle.calculate(...)`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_lammps_input.py`**: Verify `in.lammps` contains the correct `fix halt` command and hybrid pair style.

### 5.2. Integration Testing
*   **`test_watchdog_trigger.py`**:
    *   **Mock Mode**:
        *   Mock the `run_md` method to "fake" a halt event after 100 steps.
        *   Return a structure with tagged "high uncertainty".
    *   **Real Mode**:
        *   Run MD with a very low threshold (e.g., 0.1) on a dummy potential.
        *   Verify that LAMMPS actually stops early.
