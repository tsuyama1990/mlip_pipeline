# Cycle 05 Specification: Dynamics Engine & OTF Learning

## 1. Summary
Cycle 05 implements the "Dynamics Engine", the core driver for exploration. This component executes Molecular Dynamics (MD) using LAMMPS and optionally Kinetic Monte Carlo (kMC) using EON. A critical feature is "On-the-Fly (OTF) Learning," where the simulation monitors the potential's uncertainty (extrapolation grade $\gamma$) and halts if it exceeds a threshold, triggering a new learning cycle.

## 2. System Architecture

### 2.1 File Structure
**Bold** files are to be created or modified in this cycle.

```ascii
.
├── src/
│   └── mlip_autopipec/
│       ├── components/
│       │   ├── dynamics/
│       │   │   ├── **lammps_driver.py**   # LAMMPS Wrapper
│       │   │   ├── **eon_driver.py**      # EON Wrapper (kMC)
│       │   │   └── **otf_manager.py**     # Uncertainty Check Logic
│       ├── domain_models/
│       │   └── **dynamics_config.py**     # MD settings (temp, steps)
│       └── utils/
│           └── **lammps_utils.py**        # Input file generation
```

## 3. Design Architecture

### 3.1 LAMMPS Driver (`src/mlip_autopipec/components/dynamics/lammps_driver.py`)

*   **`LammpsDriver` Class**:
    *   Inherits from `BaseDynamics`.
    *   **Execution**: Can run LAMMPS via `subprocess` (using input files) or via the `lammps` Python module (library mode). We prefer the Python module if available for better control.
    *   **Input Generation**: Generates `in.lammps`, `data.lammps`.
    *   **Hybrid Potential**: Automatically writes `pair_style hybrid/overlay pace zbl` commands to ensure core repulsion stability.
    *   **OTF Monitoring**:
        *   Uses `compute pace` to get $\gamma$.
        *   Uses `fix halt` to stop if `v_max_gamma > threshold`.

### 3.2 OTF Manager (`src/mlip_autopipec/components/dynamics/otf_manager.py`)
*   Analyzes the output of a halted MD run.
*   Extracts the "bad" structure (the snapshot where $\gamma$ was high).
*   Selects a cluster of atoms around the high-uncertainty region for DFT calculation (integration with `Embedding` from Cycle 03).

### 3.3 EON Driver (`src/mlip_autopipec/components/dynamics/eon_driver.py`)
*   Wraps the EON client for kMC simulations.
*   Since EON communicates via files or sockets, this driver manages the `client` execution and parses results.
*   (Note: For the initial version, this might be a stub or a simple mock if EON installation is difficult in the development environment).

## 4. Implementation Approach

1.  **LAMMPS Input Generator**: Implement functions to write `in.lammps` with `fix npt`, `fix langevin`, and `pair_style hybrid`.
2.  **LammpsDriver**:
    *   Implement `run_md(structure, potential, config)`.
    *   Parse `log.lammps` or use Python API to detect if the run finished or halted.
    *   If halted, return the dump file and the step number.
3.  **OTF Logic**:
    *   If `LammpsDriver` returns "halted", the `Orchestrator` (Cycle 6 update, but prepared here) needs to know *what* to calculate.
    *   Implement `extract_high_uncertainty_candidates(dump_file)` in `otf_manager.py`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Input Generation**: Verify `in.lammps` contains the correct `pair_style` and `fix halt` commands.
*   **Log Parsing**: Feed a dummy LAMMPS log file with "Fix halt condition met" and verify the parser detects it.

### 5.2 Integration Testing
*   **Mock LAMMPS**: Use a `MockLammps` that simulates a run.
    *   Case A: Run completes successfully (low uncertainty).
    *   Case B: Run halts at step 500 (high uncertainty).
*   **Real LAMMPS (Local)**:
    1.  Run a short MD (100 steps) on a small system using a dummy potential.
    2.  Check if `dump.lammps` is created.
    3.  Check if `log.lammps` exists.
