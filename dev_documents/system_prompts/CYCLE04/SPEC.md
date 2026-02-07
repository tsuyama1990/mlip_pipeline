# Cycle 04 Specification: Molecular Dynamics & On-the-Fly Learning

## 1. Summary
Cycle 04 implements the "Executor" module, integrating the LAMMPS molecular dynamics engine. This is the heart of the active learning loop. We will implement the `DynamicsEngine` to run simulations using the ACE potentials trained in Cycle 03. Crucially, we introduce the "On-the-Fly" (OTF) watchdog, which monitors the extrapolation grade ($\gamma$) in real-time and halts the simulation if the potential enters an uncertain region. This prevents unphysical results and signals the Orchestrator to acquire new data.

## 2. System Architecture

### File Structure
Files to be created/modified are marked in **bold**.

```
mlip-pipeline/
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   ├── **config.py**       # Add DynamicsConfig (LAMMPS params)
│       │   └── **potential.py**    # Add ExplorationResult
│       ├── infrastructure/
│       │   ├── dynamics/
│       │   │   ├── **__init__.py**
│       │   │   ├── **lammps_driver.py** # MDInterface implementation
│       │   │   └── **otf_watchdog.py**  # Gamma monitoring logic
│       └── utils/
│           └── **lammps_utils.py** # Helper to generate in.lammps
└── tests/
    └── integration/
        └── **test_dynamics_pipeline.py**
```

## 3. Design Architecture

### Domain Models (`domain_models/`)

-   **`DynamicsConfig`**:
    -   `engine_type`: Literal["lammps", "mock"]
    -   `temperature`: float (Default: 300.0 K)
    -   `pressure`: float (Default: 0.0)
    -   `timestep`: float (Default: 0.001 ps)
    -   `steps`: int (Default: 10000)
    -   `otf_threshold`: float (Default: 5.0)

-   **`ExplorationResult`**:
    -   `final_structure`: Structure
    -   `trajectory_path`: Path
    -   `termination_reason`: Literal["MaxSteps", "UncertaintyHalt"]
    -   `max_gamma`: float
    -   `halted_step`: Optional[int]

### Infrastructure (`infrastructure/`)

-   **`MDInterface` (implements `BaseDynamics`)**:
    -   `run(potential: Path, initial_structure: Structure, config: DynamicsConfig) -> ExplorationResult`:
        -   Generates `in.lammps`.
        -   Sets up `pair_style hybrid/overlay pace zbl`.
        -   Sets up `fix halt` condition based on `v_max_gamma`.
        -   Executes LAMMPS via Python API (`lammps` module) or subprocess.
        -   Parses log to determine exit status.

## 4. Implementation Approach

1.  **LAMMPS Integration**: Use the official `lammps` Python package if available. Fallback to `subprocess` with `lmp_serial` / `lmp_mpi` if the Python bindings are missing (for robust deployment).
2.  **Hybrid Potential Generation**: Implement logic in `lammps_utils.py` to automatically write the `pair_coeff` lines for ZBL + ACE overlay. This is critical for physical robustness.
3.  **OTF Watchdog**:
    -   Use `compute pace` (part of USER-PACE package in LAMMPS).
    -   Define a variable `v_max_gamma` equal to the max gamma of all atoms.
    -   Use `fix halt` to stop if `v_max_gamma > threshold`.
4.  **Log Parsing**: Robustly parse LAMMPS log files to extract the timestep where it halted and the maximum gamma value observed.

## 5. Test Strategy

### Unit Testing (`tests/unit/`)
-   **`test_lammps_driver.py`**:
    -   Verify generation of `in.lammps` string.
    -   Check that `fix halt` command is correctly formatted with the user's threshold.
    -   Check that `pair_style hybrid/overlay` is always present.

### Integration Testing (`tests/integration/`)
-   **`test_dynamics_pipeline.py`**:
    -   Requires `lammps` with USER-PACE installed.
    -   If not available, mock the `lammps` object.
    -   Run a short MD (10 steps).
    -   Assert that `dump.lammps` is created.
    -   Assert that `ExplorationResult` correctly identifies "MaxSteps" (if potential is good) or "UncertaintyHalt" (if we force a high gamma).
