# Cycle 05 Specification: Dynamics Engine & On-the-Fly Learning

## 1. Summary

Cycle 05 is the "Closing the Loop" phase. We implement the **Dynamics Engine**, which uses the trained potential to run Molecular Dynamics (MD) simulations. This is not just a passive runner; it is an active participant in the learning process.

We implement the **On-the-Fly (OTF) Loop**. The engine uses LAMMPS to run MD while calculating the extrapolation grade ($\gamma$) for every atom at frequent intervals. If $\gamma$ exceeds a safety threshold, the simulation is **Halted**. The system then extracts the "dangerous" configuration, requests ground-truth DFT data for it, retrains the potential, and resumes the simulation.

To ensure safety during this process, we implement **Hybrid Potentials**. The ACE potential is overlaid with a ZBL/LJ baseline. This ensures that even if the ML model predicts unphysical forces in a high-uncertainty region, the physics-based repulsion prevents atomic overlap and simulation crashes.

## 2. System Architecture

```ascii
src/mlip_autopipec/
├── ...
├── dynamics/
│   ├── __init__.py
│   ├── **lammps_driver.py** # Controls LAMMPS execution
│   ├── **otf_handler.py**   # Manages Halt & Resume logic
│   └── **lammps_templates.py** # Jinja2 templates for in.lammps
└── ...
```

## 3. Design Architecture

### 3.1. LAMMPS Driver (`lammps_driver.py`)
*   **`MDInterface`**:
    *   Generates `in.lammps` from templates.
    *   Configures `pair_style hybrid/overlay pace zbl`.
    *   Sets up `compute pace` to monitor gamma.
    *   Sets up `fix halt` to stop if `max(gamma) > threshold`.

### 3.2. OTF Handler (`otf_handler.py`)
*   **`HaltEvent`**: Value object containing `timestep`, `max_gamma`, and the `snapshot`.
*   **`OTFManager`**:
    *   `run_exploration()`: Calls LAMMPS.
    *   `handle_halt()`: Logic to extract the specific atoms with high gamma and propose them for the Active Set.

## 4. Implementation Approach

1.  **Template Engine**: Create robust Jinja2 templates for LAMMPS input files. Hardcode the `hybrid/overlay` logic to ensure it's always used.
2.  **LAMMPS Integration**: Use the `lammps` Python library (recommended) or `subprocess`.
    *   If using subprocess, we must parse `log.lammps` to determine if the exit was caused by `fix halt`.
3.  **Halt Logic**:
    *   When Halt occurs, read the dump file.
    *   Filter atoms with `gamma > threshold`.
    *   (Optional) Perform local sampling (small displacement) around these atoms to probe the uncertainty well.
4.  **Orchestrator Update**: The "Exploration" phase now becomes dynamic. It's no longer just "run for 1ns"; it's "run until 1ns OR uncertainty spike".

## 5. Test Strategy

### 5.1. Unit Testing
*   **Template Test**: Render the LAMMPS template. Assert it contains `pair_style hybrid/overlay` and `fix halt`.
*   **Log Parsing**: Feed a dummy LAMMPS log containing "Fix halt condition met". Assert `detect_halt()` returns True.

### 5.2. Integration Testing
*   **Watchdog Test**:
    *   Start a LAMMPS run (Mocked or Real).
    *   Inject a high gamma value (if mocking the compute) or force a collision (if real).
    *   Verify the simulation stops early.
    *   Verify the `OTFManager` correctly identifies the final snapshot as the "Bad Structure".
