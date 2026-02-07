# Cycle 05 Specification: Dynamics & On-the-Fly (OTF) Engine

## 1. Summary
This cycle develops the "Dynamics Engine", the component responsible for exploring the configuration space via Molecular Dynamics (MD) and identifying regions where the current potential is unreliable. We interface with LAMMPS to run simulations using the ACE potential. A critical feature is the implementation of "Hybrid Potentials" (combining ACE with a physical baseline like ZBL/LJ) to ensure simulation stability even in high-energy regimes. Furthermore, we implement the "On-the-Fly" (OTF) monitoring loop, which leverages LAMMPS's `fix halt` command to automatically terminate simulations when the model's uncertainty ($\gamma$) exceeds a safety threshold, triggering the active learning refinement.

## 2. System Architecture

The following file structure will be created/modified. Files in **bold** are the primary deliverables for this cycle.

```ascii
src/
└── mlip_autopipec/
    ├── implementations/
    │   └── **dynamics/**
    │       ├── **__init__.py**
    │       ├── **lammps_engine.py**    # Main Dynamics Class
    │       └── **input_generator.py**  # LAMMPS Script Builder
    └── utils/
        └── **lammps_runner.py**        # Execution Helper
```

## 3. Design Architecture

### 3.1. Dynamics Engine
The `LammpsEngine` class implements the `BaseDynamics` interface.
-   **Configuration**: `DynamicsConfig` controls ensemble (NVT/NPT), temperature, pressure, and timesteps.
-   **Hybrid Potential**: It strictly enforces the use of `pair_style hybrid/overlay`. The input script generator automatically adds `pair_coeff * * zbl` (or LJ) alongside `pair_coeff * * pace`. This is a non-negotiable safety feature.

### 3.2. OTF Monitoring (`fix halt`)
The engine configures LAMMPS to calculate the extrapolation grade $\gamma$ at every step (or every $N$ steps).
-   Command: `fix halt_uncertainty all halt 10 v_max_gamma > ${threshold} error hard`
-   Behaviour: If the uncertainty spikes, LAMMPS crashes with a specific error message. The Python wrapper catches this error, interprets it as an "Active Learning Trigger" (not a software bug), and extracts the final frame for labelling.

## 4. Implementation Approach

### Step 1: Input Generator
Implement `input_generator.py`.
-   Create a class `LammpsInputBuilder`.
-   Method `build_hybrid_potential(elements, potential_path, zbl_params)`.
-   Method `build_md_run(temp, pressure, steps)`.
-   Method `build_uncertainty_check(threshold)`.

### Step 2: Dynamics Engine
Implement `LammpsEngine` in `lammps_engine.py`.
-   `run_exploration(potential, initial_structure) -> ExplorationResult`.
-   Write input script `in.lammps` and data file `data.lammps`.
-   Execute LAMMPS (via `subprocess` or `lammps` python module if available).
-   Handle `CalledProcessError`: check stdout for "halt" message.

### Step 3: Structure Extraction
Implement logic to parse the LAMMPS dump file (or restart file) to retrieve the atomic configuration at the moment of the halt. Identify the specific atoms with high $\gamma$ to assist the Oracle (Cycle 03) in periodic embedding.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Input Verification**: Assert that the generated `in.lammps` contains `pair_style hybrid/overlay`. Assert that `fix halt` is present with the correct threshold.
-   **Data File**: Assert that `Structure` -> `data.lammps` conversion is correct (box dimensions, atom types).

### 5.2. Integration Testing (Mocked LAMMPS)
-   Mock the execution. Simulate a "Halt" scenario by having the mock write a specific log message and exit with a non-zero code.
-   Verify that `LammpsEngine` correctly catches this, sets `result.status = 'HALTED'`, and returns the path to the dump file.
