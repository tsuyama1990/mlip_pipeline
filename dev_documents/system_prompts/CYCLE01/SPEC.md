# Cycle 01: Core Framework & Oracle

## 1. Summary

Cycle 01 lays the foundation for the PyAcemaker system. The primary objective is to establish the project structure, implement the robust configuration management system using Pydantic, and create the "Oracle" module capable of running Density Functional Theory (DFT) calculations. At the end of this cycle, the system will be able to read a user-provided configuration file (YAML), validate it, and execute a "static" DFT calculation (Single Point Energy & Forces) using Quantum Espresso (QE) on a given atomic structure. This cycle does not yet involve machine learning or molecular dynamics, but it builds the critical infrastructure for ground-truth data generation which is the bottleneck of any MLIP project.

We will focus on the `Orchestrator`'s initialization logic and the `DFTManager` which handles the complexity of interfacing with external QE executables. We will also define the strict Pydantic schemas that will govern the data flow throughout the project's lifecycle, ensuring type safety and fail-fast validation.

## 2. System Architecture

The architecture for Cycle 01 focuses on the `config`, `orchestrator`, and `dft` modules.

### File Structure

**mlip_autopipec/**
├── **app.py**                  # CLI Entry point (Typer)
├── **config/**
│   ├── **__init__.py**
│   ├── **main_config.py**      # Root WorkflowConfig
│   └── **schemas/**
│       ├── **__init__.py**
│       └── **dft.py**          # DFTConfig schema
├── **orchestrator/**
│   ├── **__init__.py**
│   └── **runner.py**           # Main execution logic (skeleton)
└── **dft/**
    ├── **__init__.py**
    ├── **manager.py**          # DFTManager class
    └── **qe_runner.py**        # QERunner class (subprocess wrapper)

### Component Description

*   **`app.py`**: The command-line interface. It uses `typer` to accept arguments like `--config config.yaml` and initiates the `Orchestrator`.
*   **`config/`**: Contains the Pydantic models. `DFTConfig` defines parameters like `ecutwfc`, `kspacing`, `smearing`, and `pseudopotential_dir`.
*   **`orchestrator/runner.py`**: Holds the `ProjectManager` class which loads the config and instantiates the `DFTManager`.
*   **`dft/qe_runner.py`**: A low-level wrapper around the `pw.x` executable. It handles the writing of `pw.in` files (converting ASE Atoms to QE format) and parsing of standard output/XML for energy, forces, and stress.
*   **`dft/manager.py`**: A higher-level controller. It receives a list of structures, dispatches them to `QERunner`, and handles basic error checking (e.g., if SCF fails to converge).

## 3. Design Architecture

This system relies on **Pydantic V2** for all data modeling to ensure strict validation.

### Domain Models

**`DFTConfig`** (in `config/schemas/dft.py`)
*   **Role**: Defines how DFT calculations should be run.
*   **Fields**:
    *   `command`: `str` (e.g., "mpirun -np 4 pw.x") - Validated to ensure it's safe.
    *   `pseudopotential_dir`: `DirectoryPath` - Must exist.
    *   `pseudopotentials`: `Dict[str, str]` - Mapping of Element -> Filename.
    *   `ecutwfc`: `float` - Plane wave cutoff (Ry).
    *   `kspacing`: `float` - K-point density (1/Å) for automatic grid generation.
    *   `scf_max_steps`: `int` (default 100).
    *   `mixing_beta`: `float` (default 0.7).

**`DFTResult`** (in `dft/manager.py` or separate schema)
*   **Role**: Standardized output from the Oracle.
*   **Fields**:
    *   `energy`: `float` (eV).
    *   `forces`: `List[List[float]]` (eV/Å, shape N x 3).
    *   `stress`: `List[List[float]]` (Voigt or 3x3 matrix, kBar/GPa).
    *   `converged`: `bool`.
    *   `meta`: `Dict` (runtime, number of steps).

### Key Invariants & Constraints
1.  **Safety**: The system must check that the `pw.x` command does not contain shell injection characters.
2.  **Completeness**: A `DFTResult` is only valid if `converged` is True. If False, the other fields must be ignored or marked as invalid.
3.  **Atom Mapping**: The order of atoms in the `forces` array must exactly match the input `Atoms` object.

## 4. Implementation Approach

1.  **Project Setup**:
    *   Initialize the `src/mlip_autopipec` package structure.
    *   Configure `pyproject.toml` dependencies (`ase`, `pydantic`, `typer`).

2.  **Configuration Schema Implementation**:
    *   Implement `DFTConfig` and `WorkflowConfig`.
    *   Write tests to verify that invalid YAML files raise descriptive errors.

3.  **QERunner Implementation**:
    *   Implement `write_input()`: Use `ase.io.espresso` or custom logic to write `pw.in`.
    *   Implement `run()`: Use `subprocess.run` to execute the command.
    *   Implement `parse_output()`: Read the output file to extract Energy/Forces.
    *   **Note**: Ensure `tprnfor=.true.` and `tstress=.true.` are always injected into the input.

4.  **DFTManager Implementation**:
    *   Implement `run_batch(structures: List[Atoms])`.
    *   Add basic error handling (try/except blocks for subprocess failures).

5.  **CLI Integration**:
    *   Create a simple command `mlip-auto test-dft --config config.yaml --structure structure.xyz` to verify the pipeline manually.

## 5. Test Strategy

### Unit Testing
*   **Config**: Test `DFTConfig` with valid and invalid dictionaries. Check that `kspacing` > 0, `ecutwfc` > 0.
*   **QERunner (Mocked)**:
    *   Mock `subprocess.run` to avoid running actual Quantum Espresso.
    *   Provide sample `pw.out` files (text fixtures) to test the parsing logic (`parse_output`).
    *   Verify that `write_input` correctly formats the `ATOMIC_POSITIONS` and `CELL_PARAMETERS`.

### Integration Testing
*   **Real Execution (Optional/Local)**: If `pw.x` is available in the environment, run a tiny calculation (e.g., H2 molecule).
*   **ASE Integration**: Verify that `ase.Atoms` objects are correctly converted to QE inputs and that the returned `forces` array matches the shape of the input atoms.
*   **Error Handling**: Simulate a "Crash" (exit code != 0) from the mock subprocess and verify that `DFTManager` captures it and returns a `DFTResult` with `converged=False`.
