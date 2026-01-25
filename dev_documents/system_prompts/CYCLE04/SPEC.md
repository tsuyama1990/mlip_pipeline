# Cycle 04 Specification: Dynamics Phase I (Basic MD)

## 1. Summary

Cycle 04 introduces the **Dynamics Engine**, specifically interfacing with **LAMMPS**. This engine is the "Explorer" of the active learning loop, running Molecular Dynamics (MD) simulations to sample the phase space.

In this first phase of the Dynamics implementation, we focus on the ability to **run stable simulations** using the ACE potentials generated in Cycle 03. A critical requirement here is the implementation of the **Hybrid Potential** strategy (`pair_style hybrid/overlay`). This overlays a ZBL or Lennard-Jones baseline on top of the ML potential to prevent physical collapse (atoms overlapping) in regions where the ML potential is undefined (extrapolation).

We will implement the logic to generate LAMMPS input files, execute the binary, and parse the thermodynamic output (Temperature, Pressure, Density). We do not yet handle the "Active Learning / Halt" logic (reserved for Cycle 05).

## 2. System Architecture

We populate the `dynamics/` directory.

### 2.1 File Structure

```ascii
src/mlip_autopipec/
├── config/
│   └── schemas/
│       └── **inference.py**        # MD settings (Temp, Pressure, Steps)
├── **dynamics/**
│   ├── **__init__.py**
│   ├── **lammps.py**               # LAMMPS Runner
│   ├── **writer.py**               # Input File Generator
│   └── **parser.py**               # Log/Dump Parser
└── orchestration/
    └── phases/
        └── **exploration.py**      # Exploration Phase Logic
```

## 3. Design Architecture

### 3.1 LAMMPS Input Writer (`dynamics/writer.py`)

*   **Responsibility**: Convert `InferenceConfig` and `Structure` into a valid `in.lammps` file.
*   **Key Logic**:
    *   **Hybrid Pair Style**: Must generate:
        ```lammps
        pair_style hybrid/overlay pace zbl 1.0 2.0
        pair_coeff * * pace potential.yace Element1 Element2
        pair_coeff * * zbl 14 6
        ```
    *   **Ensembles**: Support `fix nvt` and `fix npt` based on config.

### 3.2 LAMMPS Runner (`dynamics/lammps.py`)

*   **Responsibility**: Execute LAMMPS.
*   **Input**: `Atoms` object (initial structure), `potential_path`.
*   **Output**: `final_structure` (Atoms), `trajectory_path`.
*   **Method**: `run_md(atoms, config) -> SimulationResult`.

## 4. Implementation Approach

1.  **Step 1: Input Writer Implementation.**
    *   Create a class that takes an ASE Atoms object and writes `data.lammps`.
    *   Implement string templates for `in.lammps`.
    *   Implement the ZBL parameter lookups (atomic numbers).

2.  **Step 2: Runner Implementation.**
    *   Use `subprocess.Popen` to run LAMMPS. We need `Popen` (not just `run`) because in the next cycle we might want to monitor stdout in real-time (though for now `run` is fine).
    *   Ensure `potential.yace` is copied or linked to the execution directory.

3.  **Step 3: Output Parsing.**
    *   Implement a parser for `log.lammps` to extract thermodynamic data (to check for stability, e.g., if Temp exploded).
    *   Use `ase.io.read` to load the final structure from the dump file.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Input File Verification:**
    *   Create a test that generates an input file for a Ti-O system.
    *   Assert that `pair_style hybrid/overlay` is present.
    *   Assert that `pair_coeff * * zbl` line contains correct atomic numbers (22 and 8).
    *   Assert that `fix nvt` parameters match the config (Temp=300K).

### 5.2 Integration Testing
*   **Mock Execution:**
    *   Mock the `lmp` executable.
    *   The test should check that the runner creates a run directory, places `in.lammps`, `data.lammps`, and `potential.yace` there, and tries to execute the command.
*   **ASE Connectivity:**
    *   Verify that an input `Atoms` object is correctly converted to `data.lammps` format (checking box dimensions).
