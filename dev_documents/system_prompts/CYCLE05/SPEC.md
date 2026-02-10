# Cycle 05: Dynamics Engine (MD) & Hybrid Potential

## 1. Summary

Cycle 05 implements the "Dynamics Engine," which is responsible for running Molecular Dynamics (MD) simulations using the LAMMPS software. This module is the "Explorer" in the Active Learning loop.

Crucially, we must address the risk of potential failure in extrapolation regions. To do this, we implement two key safety mechanisms:
1.  **Hybrid Potential**: We will use LAMMPS's `pair_style hybrid/overlay` to combine the machine-learned ACE potential with a robust physics-based baseline (ZBL or LJ). This ensures that even if the ACE potential predicts non-physical forces at short distances, the baseline repulsion will prevent atoms from collapsing (nuclear fusion).
2.  **Uncertainty Watchdog**: We will use the `compute pace` command (from the USER-PACE package) to calculate the extrapolation grade ($\gamma$) for every atom at every step. A `fix halt` command will be configured to automatically stop the simulation if $\gamma$ exceeds a safety threshold (e.g., 5.0).

## 2. System Architecture

The following file structure will be modified/created. Files in **bold** are the primary deliverables for this cycle.

```
.
├── config.yaml
├── src/
│   └── mlip_autopipec/
│       ├── core/
│       │   ├── config.py         # Update DynamicsConfig
│       ├── components/
│       │   ├── base.py
│       │   └── **dynamics.py**   # LAMMPS Dynamics Implementation
│       └── utils/
│           └── **lammps_driver.py** # LAMMPS I/O
└── tests/
    ├── **test_dynamics.py**
    └── **test_lammps_driver.py**
```

## 3. Design Architecture

### LAMMPS Dynamics (`components/dynamics.py`)
The `LAMMPSDynamics` implements the `BaseDynamics` interface.
*   `explore(potential) -> ExplorationResult`:
    1.  Generates a LAMMPS input script (`in.lammps`).
    2.  Writes the initial structure (`data.lammps`).
    3.  Runs LAMMPS via `subprocess` or `lammps` python module.
    4.  Parses the output log to determine if the run completed or halted.
    5.  If halted, identifies the snapshot with the highest uncertainty.

### Hybrid Potential Logic
The input script generator must support:
```lammps
pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace potential.yace Element1 Element2
pair_coeff * * zbl 14 14  # Si-Si repulsion
```
This requires `DynamicsConfig` to specify the baseline potential type and parameters.

### Uncertainty Watchdog Logic
The input script must include:
```lammps
compute gamma all pace potential.yace ... gamma_mode=1
variable max_gamma equal max(c_gamma)
fix watchdog all halt 10 v_max_gamma > 5.0 error hard
```

### LAMMPS Driver (`utils/lammps_driver.py`)
*   `write_lammps_data(atoms, filename)`: Writes the `data.lammps` file.
*   `run_lammps(input_file, log_file)`: Executes the binary.
*   `parse_log(log_file)`: Reads the log to find "Halted" messages and extract the final step number.

## 4. Implementation Approach

1.  **Driver Implementation**: Implement `utils/lammps_driver.py`. Use `ase.io.lammpsdata.write_lammps_data` for structure writing.
2.  **Dynamics Implementation**: Implement `components/dynamics.py`.
3.  **Input Template**: Create a robust template for `in.lammps` using `jinja2` or string formatting. Include sections for: Init, Atom Definition, Potential (Hybrid), Minimize, Equilibration (NVT/NPT), and Production (with Watchdog).
4.  **Configuration**: Update `DynamicsConfig` to include `timestep`, `temperature`, `pressure`, `hybrid_potential` settings.
5.  **Integration**: Update `Orchestrator` to use `LAMMPSDynamics`.

## 5. Test Strategy

### Unit Testing
*   **Input Script Generation**:
    *   Configure `LAMMPSDynamics` with `hybrid=True`.
    *   Call `_generate_input_script()`.
    *   Assert that `pair_style hybrid/overlay` is present.
    *   Assert that `fix watchdog` is present with the correct threshold.

### Integration Testing (with Mock LAMMPS)
*   **Halt Detection**:
    *   Create a mock log file containing "ERROR: Halted by fix halt".
    *   Call `parse_log()`.
    *   Assert that it returns `halted=True` and the correct step number.

### Integration Testing (Real LAMMPS - Optional)
*   **Short Run**:
    *   Run a 100-step MD on a small cell.
    *   Verify that `log.lammps` is created.
    *   Verify that the run completes successfully (if potential is good) or halts (if potential is bad/mocked to be bad).
