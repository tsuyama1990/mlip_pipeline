# Cycle 02: Basic Exploration (The One-Shot Pipeline)

## 1. Summary

With the foundation laid in Cycle 01, Cycle 02 moves to the first phase of the active learning loop: **Exploration**. In a mature system, this involves complex adaptive policies and uncertainty-driven halting. However, before running, we must walk. The goal of this cycle is to establish the capability to drive an external Molecular Dynamics engine (LAMMPS) from Python, execute a simulation, and parse the results back into our domain models.

We refer to this as the "One-Shot Pipeline". It is not yet a loop. It is a linear execution: Generate a structure -> Run MD -> Parse Output. This seemingly simple sequence involves significant complexity:
1.  **Input Generation**: Converting our `Structure` object into a LAMMPS data file.
2.  **Script Generation**: Writing a valid `in.lammps` script that defines interatomic potentials (initially Lennard-Jones for testing), thermodynamic ensembles (NVT/NPT), and output dumps.
3.  **Execution Management**: Launching the LAMMPS binary via `subprocess`, managing standard output/error streams, and handling timeouts or crashes.
4.  **Output Parsing**: Reading the binary or text dump files produced by LAMMPS and converting them back into a trajectory of `Structure` objects.

By the end of this cycle, we will have a `LammpsRunner` (implemented via IO helpers) that serves as the interface between our Python world and the HPC world. We will also implement a basic `StructureGenerator` that can generate random supercells to feed into this runner. This proves that we can "touch" the physics engine and retrieve data, a prerequisite for the more complex Oracle (Cycle 03) and Training (Cycle 04) phases.

## 2. System Architecture

In this cycle, we introduce the `modules` package and expand `infrastructure`. The architecture shifts from static data definitions to dynamic execution flows.

### File Structure
Files to be created/modified are in **bold**.

```ascii
mlip_autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   ├── **structure.py**        # Updated with Job models
│       │   └── **config.py**           # Updated with Lammps/StructureGen config
│       ├── infrastructure/
│       │   ├── **io.py**               # LAMMPS IO and Subprocess helpers
│       │   └── logging.py
│       ├── modules/
│       │   └── structure_gen/
│       │       ├── **generator.py**    # StructureGenerator class
│       │       └── **strategies.py**   # Generation strategies (Bulk, Rattle)
│       └── cli/
│           └── **commands.py**         # CLI command to run one-shot
└── tests/
    ├── domain_models/
    │   ├── **test_structure.py**
    │   └── **test_config.py**
    ├── infrastructure/
    │   └── **test_io.py**
    ├── unit/
    │   └── **test_structure_gen.py**
    └── e2e/
        └── **test_exploration_phase.py**
```

### Component Interaction (The "One-Shot" Flow)

1.  **Orchestrator (CLI `commands.py`)**:
    -   Instantiated by the CLI command `run-cycle-02`.
    -   Loads `Config`.
    -   Calls `StructureGenerator` to get an initial atomic configuration (e.g., Bulk Silicon).

2.  **StructureGenerator (`generator.py`)**:
    -   Uses `strategies.py` (wrapping `ase.build`) to create a perfect crystal.
    -   Applies `rattle` (random displacement) to break symmetry.
    -   Returns a `Structure` object.

3.  **Lammps Interaction (`infrastructure/io.py`)**:
    -   **Input**: Receives the `Structure` and parameters.
    -   **Preparation**:
        -   Writes `data.lammps` (atomic coordinates) using `write_lammps_data`.
        -   Writes `in.lammps` (commands).
    -   **Execution**:
        -   Calls `mpirun -np 4 lmp_serial -in in.lammps` (or similar) via `run_subprocess`.
        -   Captures `stdout` and `stderr`.
    -   **Parsing**:
        -   Reads `dump.lammpstrj` using `read_lammps_dump`.
        -   Extracts the final frame.
    -   **Output**: Returns a `LammpsResult` containing the final `Structure` and status.

## 3. Design Architecture

### 3.1. Job Domain Model (`domain_models/structure.py`)
We need a generic way to represent an external calculation. We define these in `structure.py` (or `__init__.py`) alongside the `Structure` definition.

-   **Enum `JobStatus`**:
    -   `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `TIMEOUT`.

-   **Class `JobResult`** (Base Model):
    -   `job_id`: `str` (UUID).
    -   `status`: `JobStatus`.
    -   `work_dir`: `Path`.
    -   `duration_seconds`: `float`.
    -   `log_content`: `str` (Tail of the log).

-   **Class `LammpsResult`** (Inherits `JobResult`):
    -   `final_structure`: `Structure`.
    -   `trajectory_path`: `Path`.

### 3.2. Configuration (`domain_models/config.py`)
-   **Class `LammpsConfig`**:
    -   `command`: `str` (e.g. "lmp_serial").
    -   `timeout`: `float` (seconds).
-   **Class `StructureGenConfig`**:
    -   `element`: `str`.
    -   `crystal_structure`: `str` (e.g. "fcc").
    -   `lattice_constant`: `float`.
    -   `supercell`: `tuple[int, int, int]`.

### 3.3. Structure Generator (`modules/structure_gen/generator.py`)
-   **Class `StructureGenerator`**:
    -   Uses `strategies.py` to implement generation logic.
    -   `generate(config: StructureGenConfig) -> Structure`.

## 4. Implementation Approach

### Step 1: Job Models & Config
-   Update `src/mlip_autopipec/domain_models/structure.py`.
-   Update `src/mlip_autopipec/domain_models/config.py`.

### Step 2: Infrastructure IO
-   Implement `write_lammps_data`, `read_lammps_dump`, `run_subprocess` in `infrastructure/io.py`.

### Step 3: Structure Generator
-   Implement `modules/structure_gen/strategies.py` and `generator.py`.

### Step 4: Orchestration
-   Implement `run_cycle_02` in `cli/commands.py` which ties it all together.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Structure Generator**: Test `generate`.
-   **IO**: Mock `subprocess.run` to test `run_subprocess`.

### 5.2. Integration Testing (Local)
-   Run a full cycle using `run-cycle-02`.
