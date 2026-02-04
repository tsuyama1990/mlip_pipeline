# CYCLE 05 Specification: Dynamics Engine - MD & OTF

## 1. Summary

Cycle 05 implements the "Dynamics Engine", specifically the Molecular Dynamics (MD) capability using LAMMPS. This module is the heart of the "Active Learning" process. It runs MD simulations using the trained potential, but with a critical safety mechanism: the **Uncertainty Watchdog**. By monitoring the extrapolation grade ($\gamma$) in real-time, the system can stop the simulation *before* it becomes unphysical, extract the problematic atomic configuration, and request ground-truth data from the Oracle. This cycle also enforces the "Hybrid Potential" architecture, ensuring that the ACE potential is always overlaid with a ZBL/LJ baseline for physical robustness.

## 2. System Architecture

```ascii
src/mlip_autopipec/
├── services/
│   ├── external/
│   │   └── lammps_interface.py    # [CREATE] Wrapper for LAMMPS
│   └── dynamics/
│       ├── __init__.py
│       ├── md_engine.py           # [CREATE] LammpsMD implementation
│       └── otf_handler.py         # [CREATE] On-The-Fly Logic (Halt & Diagnose)
```

## 3. Design Architecture

### LAMMPS Interface (`lammps_interface.py`)
-   **Role**: Generates `in.lammps` scripts and manages execution.
-   **Key Feature**: `generate_hybrid_potential_block()`. This writes:
    ```lammps
    pair_style hybrid/overlay pace zbl 1.0 2.0
    pair_coeff * * pace potential.yace ...
    pair_coeff * * zbl ...
    ```
-   **Watchdog**: Adds `compute pace_gamma` and `fix halt` commands to the script.

### OTF Handler (`otf_handler.py`)
-   **Role**: Logic for processing halted simulations.
-   **Input**: A directory with a halted LAMMPS run.
-   **Output**: A list of `StructureMetadata` (the "high uncertainty" snapshots).
-   **Algorithm**:
    1.  Parse LAMMPS log/output to find the timestep where halt occurred.
    2.  Read the dump file corresponding to that timestep.
    3.  Identify atoms with high $\gamma$.
    4.  Extract a cluster around those atoms (using Cycle 03 Embedding logic).

## 4. Implementation Approach

1.  **Implement `LammpsMD`**:
    -   Use `subprocess` to call `lmp` (or `lammps` python module if available/preferred).
    -   Must support template-based `in.lammps` generation.

2.  **Implement Watchdog Logic**:
    -   Ensure `compute pace` is correctly configured to calculate gamma.
    -   Ensure `fix halt` is set to trigger on `v_max_gamma > threshold`.

3.  **Implement OTF Extraction**:
    -   Write a parser for LAMMPS dump files (using `ase.io.read`).
    -   Integrate with `PeriodEmbedding` from Cycle 03 to create trainable structures.

## 5. Test Strategy

### Unit Testing
-   **Input Generation**: Verify that the generated `in.lammps` contains the correct `pair_style hybrid/overlay` and `fix halt` commands.
-   **Log Parsing**: Create a dummy LAMMPS log file indicating a halt at step 500. Verify the parser correctly identifies the step.

### Integration Testing
-   **Mock Execution**: Simulate a "Halt" scenario by providing a fake dump file with high-gamma atoms and a fake exit code from the subprocess. Verify that the `OTFHandler` extracts the correct structure.
