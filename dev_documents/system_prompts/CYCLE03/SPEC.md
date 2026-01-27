# Cycle 03: Dynamics Engine (LAMMPS MD) & Uncertainty

## 1. Summary

Cycle 03 focuses on the "Inference" and "Exploration" capabilities. We implement the `Dynamics Engine`, specifically interfacing with the LAMMPS molecular dynamics software. The critical innovation here is the implementation of "Physics-Informed Robustness" via Hybrid Potentials (ACE + ZBL/LJ) and "Active Learning" triggers via Uncertainty Monitoring.

We will build the `LammpsRunner` which can construct input files that overlay the machine learning potential with a repulsive baseline to prevent simulation crashes. Furthermore, we will implement the "Watchdog" mechanism using `fix halt` in LAMMPS, which stops the simulation the moment the model's uncertainty ($\gamma$) exceeds a safety threshold. This mechanism is the trigger for the active learning loop.

## 2. System Architecture

### File Structure

**mlip_autopipec/**
├── **dynamics/**
│   ├── **__init__.py**
│   └── **lammps/**
│       ├── **__init__.py**
│       ├── **runner.py**       # LammpsRunner class
│       ├── **inputs.py**       # Input file generation logic
│       ├── **log_parser.py**   # Log file analysis
│       └── **potential.py**    # Hybrid potential configuration
└── **inference/**
    ├── **__init__.py**
    └── **config.py**           # InferenceConfig schema

### Component Description

*   **`dynamics/lammps/runner.py`**: Manages the execution of LAMMPS. It handles `stdin` / `stdout` piping and environment setup.
*   **`dynamics/lammps/inputs.py`**: Responsible for writing `in.lammps`. It translates high-level Python commands (e.g., "Run NPT at 300K") into LAMMPS script syntax.
*   **`dynamics/lammps/potential.py`**: Generates the complex `pair_style hybrid/overlay` commands. It ensures that the ACE potential (`pace`) is correctly combined with a ZBL or LJ baseline to handle short-range repulsion.
*   **`dynamics/lammps/log_parser.py`**: Reads `log.lammps` to determine if the simulation finished normally or was halted by the watchdog. It extracts the final step number and the maximum $\gamma$ value.

## 3. Design Architecture

### Domain Models

**`InferenceConfig`** (in `config/schemas/inference.py`)
*   **Role**: Defines MD simulation parameters.
*   **Fields**:
    *   `ensemble`: `Literal["nvt", "npt"]`
    *   `temperature`: `float`
    *   `pressure`: `float`
    *   `timestep`: `float` (default 0.001 ps)
    *   `steps`: `int`
    *   `uncertainty_threshold`: `float` (default 5.0) - The $\gamma$ limit.

**`HaltStatus`**
*   **Role**: Return object from the Runner.
*   **Fields**:
    *   `completed`: `bool`
    *   `halted`: `bool`
    *   `final_step`: `int`
    *   `max_gamma`: `float`
    *   `reason`: `str` (e.g., "Max steps reached", "Uncertainty limit exceeded").

### Key Invariants
1.  **Safety First**: Every MD run MUST include the Hybrid/Overlay baseline. A pure ACE run is prohibited in the exploration phase to prevent segmentation faults.
2.  **Explicit Watchdog**: Every exploration MD run MUST include the `fix halt` command monitoring `c_pace_gamma`.
3.  **Element Mapping**: The order of elements in `pair_coeff` must match the `potential.yace` file and the `data.lammps` file.

## 4. Implementation Approach

1.  **Input Writer**:
    *   Implement `LammpsInputWriter` class.
    *   Method `write_structure()`: Converts ASE Atoms to `data.lammps`.
    *   Method `write_potential()`: Generates:
        ```lammps
        pair_style hybrid/overlay pace zbl 1.0 2.0
        pair_coeff * * pace potential.yace ...
        pair_coeff * * zbl ...
        ```
    *   Method `write_md_settings()`: Sets `fix npt`, `timestep`, etc.
    *   Method `write_watchdog()`: Generates:
        ```lammps
        compute gamma all pace ...
        fix stopper all halt 10 v_max_gamma > 5.0 error hard
        ```

2.  **Lammps Runner**:
    *   Implement `run(structure, potential_path, config)`.
    *   Create a temporary directory.
    *   Write `in.lammps`, `data.lammps`, copy `potential.yace`.
    *   Execute `lmp_serial < in.lammps`.
    *   Catch `subprocess.CalledProcessError`. Since we use `error hard` in `fix halt`, LAMMPS will exit with non-zero code. This is an **expected behavior** for active learning, not a system crash. The Runner must catch this and verify via the log parser if it was a "good halt".

3.  **Log Parser**:
    *   Regex parsing of the log file to find "Fix halt condition met" or similar messages.

## 5. Test Strategy

### Unit Testing
*   **Input Generation**: Verify the string content of `in.lammps`. Check that `pair_style hybrid/overlay` is present. Check that the element list in `pair_coeff` is correct.
*   **Log Parsing**: Feed sample log files (one successful, one halted, one crashed) to `LogParser` and verify the returned `HaltStatus`.

### Integration Testing
*   **Mocked Execution**:
    *   Since we might not have LAMMPS in the CI, we can mock the `subprocess.run` to return a specific exit code and write a dummy `log.lammps` containing "Fix halt condition met".
    *   Verify that `LammpsRunner` correctly interprets this as `halted=True` rather than raising an exception.
*   **Real Execution (Local)**:
    *   Run a short MD with a dummy potential. Force a high $\gamma$ (if possible to mock) or just check that the simulation runs.
