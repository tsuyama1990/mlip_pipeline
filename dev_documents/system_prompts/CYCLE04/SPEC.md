# Cycle 04 Specification: Dynamics Module (LAMMPS Integration)

## 1. Summary
This cycle implements the **Explorer** component using **LAMMPS** (Large-scale Atomic/Molecular Massively Parallel Simulator). The Explorer is responsible for sampling the configuration space to find new, relevant structures for the training set. In this cycle, we focus on the core capability: executing Molecular Dynamics (MD) simulations using the Hybrid Potentials (ACE + ZBL/LJ) defined in Cycle 3. We will implement the `LammpsDynamics` adapter, which dynamically generates input scripts (`in.lammps`) and parses the resulting trajectories (`dump` files).

## 2. System Architecture

### 2.1. File Structure

```
mlip-pipeline/
├── src/
│   └── mlip_autopipec/
│       ├── config/
│       │   └── **config_model.py**     # [MODIFY] Add ExplorerConfig
│       ├── infrastructure/
│       │   ├── **lammps/**
│       │   │   ├── **__init__.py**
│       │   │   ├── **adapter.py**      # [NEW] LammpsDynamics implementation
│       │   │   └── **templates.py**    # [NEW] Jinja2 templates for in.lammps
│       └── utils/
│           └── **parsing.py**          # [NEW] Utilities to parse LAMMPS dumps
└── tests/
    └── unit/
        └── **test_dynamics.py**        # [NEW] Tests for LammpsDynamics
```

## 3. Design Architecture

### 3.1. `ExplorerConfig` (Pydantic)
*   `timestep`: float (fs, e.g., 1.0).
*   `temperature`: float (K).
*   `pressure`: float (bar).
*   `n_steps`: int.
*   `sampling_interval`: int (Save structure every N steps).
*   `command`: str (e.g., `lmp_serial` or `mpirun lmp_mpi`).

### 3.2. `LammpsDynamics` Class
Implements `BaseExplorer`.
*   **Responsibilities**:
    1.  `setup_simulation()`: create working directory and input files.
    2.  `generate_input_script()`: Create `in.lammps` using templates.
    3.  **Hybrid Potential Logic**: CRITICAL. It must write:
        ```lammps
        pair_style hybrid/overlay pace zbl 1.0 2.0
        pair_coeff * * pace potential.yace Element1 Element2
        pair_coeff * * zbl 1 2 ...
        ```
    4.  `run()`: Execute LAMMPS.
    5.  `parse_output()`: Read `dump.lammps` and return list of `ase.Atoms`.

### 3.3. Input Generation Strategy
We will use **Jinja2** templates (or Python formatted strings) to generate robust LAMMPS scripts. This allows us to handle variable numbers of elements and different ensembles (NVT, NPT) flexibly.

## 4. Implementation Approach

1.  **Templates**: Create `infrastructure/lammps/templates.py`. Define templates for:
    *   Initialization (units metal, atom_style atomic).
    *   Potential definition (Hybrid ACE+ZBL).
    *   MD Loop (fix nvt/npt).
    *   Output (dump custom).
2.  **Implement Adapter**: Create `infrastructure/lammps/adapter.py`.
    *   Resolve element masses and atomic numbers (for ZBL).
    *   Construct the command line.
    *   Execute via `subprocess` (simpler than `lammps-python` for stability in containers).
3.  **Parsing Logic**: Implement a lightweight parser for LAMMPS custom dump files in `utils/parsing.py` (or use `ase.io.read`, but a custom parser might be faster for specific needs like reading uncertainty).
4.  **Integration**: Update `main.py` to use `LammpsDynamics`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Script Generation**:
    *   Instantiate `LammpsDynamics` with a specific config (e.g., Fe-Pt, 600K).
    *   Call `generate_input_script()`.
    *   **Assert**: `pair_style hybrid/overlay` is present.
    *   **Assert**: `fix nvt` contains correct temperature.
    *   **Assert**: `dump` command is correct.

### 5.2. Integration Testing (Mocked Binary)
*   **Mocking LAMMPS**: Create a dummy script that acts as `lmp_serial`.
    *   It should read `in.lammps`.
    *   It should write a dummy `log.lammps` and `dump.lammps`.
*   **Output Parsing**:
    *   Feed a real (or handcrafted) `dump.lammps` file to `parse_output()`.
    *   **Assert**: Returns correct number of `Atoms` objects.
    *   **Assert**: Positions and cell dimensions are correct.
