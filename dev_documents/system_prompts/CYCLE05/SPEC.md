# Cycle 05 Specification: Dynamics Engine (MD/kMC) & OTF

## 1. Summary

Cycle 05 implements the **Dynamics Engine**, the system's "arms" that explore the potential energy surface. We integrate **LAMMPS** for Molecular Dynamics (MD) and **EON** for Adaptive Kinetic Monte Carlo (aKMC). A key innovation here is the **On-the-Fly (OTF) Learning Loop**: the MD simulation constantly monitors the potential's uncertainty ($\gamma$) and automatically halts if it enters a dangerous, unknown region. This triggers a data acquisition cycle, preventing the "trash-in, trash-out" problem. We also strictly enforce **Hybrid Potentials** (ACE + ZBL/LJ) to ensure physical robustness during high-energy events.

## 2. System Architecture

Files in **bold** are the focus of this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── **lammps.py**       # LAMMPS wrapper
│   │   ├── **eon.py**          # EON (kMC) wrapper
│   │   └── **otf.py**          # Uncertainty monitoring helpers
├── domain_models/
│   ├── **config.py**           # Add LammpsConfig, EonConfig
```

## 3. Design Architecture

### 3.1. LammpsDynamics (`components/dynamics/lammps.py`)
Inherits from `BaseDynamics`.
-   **Config**: `timestep`, `temperature`, `nsteps`, `uncertainty_threshold`.
-   **Method `run(potential)`**:
    1.  **Input Generation**: Writes `in.lammps`.
        -   **Crucial**: Generates `pair_style hybrid/overlay pace zbl` commands.
        -   **Watchdog**: Adds `compute gamma ...` and `fix halt ... v_max_gamma > threshold`.
    2.  **Execution**: Runs `lmp -in in.lammps` via subprocess or library interface.
    3.  **Halt Handling**: Checks the exit code. If halted, identifies the snapshot with high uncertainty and returns it as a candidate for the Oracle.

### 3.2. EON (kMC) Wrapper (`components/dynamics/eon.py`)
-   **Role**: Explore long-time scale events.
-   **Integration**: EON calls an external script to get forces. We implement this driver script (`pace_driver.py`) which also checks $\gamma$.
-   **Logic**: If $\gamma >$ threshold during a saddle search, the driver exits with a specific code, signaling EON to abort and the Orchestrator to relearn.

## 4. Implementation Approach

1.  **Update Config**: Add `LammpsConfig` and `EonConfig`.
2.  **Implement LammpsDynamics**:
    -   Must support `ase.io.write(..., format='lammps-data')`.
    -   Implement the logic to parse `log.lammps` to find the exact timestep of failure.
3.  **Implement OTF Logic**:
    -   Create the logic to extract the specific atoms with high $\gamma$ from the dump file.
4.  **Mocking**: Create `MockLammps` that can be instructed to "fail at step 500 with gamma=10.0" to test the Orchestrator's response.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Input Generation**: Verify `in.lammps` contains `pair_style hybrid/overlay` and `fix halt`.
-   **Parser**: Feed a sample `log.lammps` with a halt event and verify it extracts the correct timestep.

### 5.2. Integration Testing
-   **Scenario**: "Safety Stop"
-   **Config**: `dynamics: lammps`, `uncertainty_threshold: 5.0`.
-   **Mock**: `MockLammps` simulates a run that hits $\gamma=6.0$.
-   **Check**: The `run()` method returns a list containing the high-uncertainty structure. The Orchestrator (from Cycle 01/02) should receive this and pass it to the Oracle.
