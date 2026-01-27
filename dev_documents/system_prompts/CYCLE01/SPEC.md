# Cycle 01 Specification: Core Framework & Oracle (DFT)

## 1. Summary

Cycle 01 focuses on establishing the foundational infrastructure of PyAcemaker. The primary goal is to build the "Oracle" component, capable of executing Density Functional Theory (DFT) calculations reliable. This cycle does not involve machine learning or active learning loops yet; it strictly ensures that we can programmatically control the ground-truth generation engine (Quantum Espresso). This involves setting up the Python project structure, defining the configuration schemas using Pydantic, and implementing a robust wrapper around the DFT code that can handle input generation and basic error recovery.

## 2. System Architecture

The following file structure will be implemented. Bold files are the primary focus of this cycle.

```ascii
mlip_autopipec/
├── **config/**
│   ├── **__init__.py**
│   ├── **loader.py**          # YAML loader with validation
│   └── **schemas/**
│       ├── **__init__.py**
│       ├── **common.py**      # Common types (Path, Element)
│       └── **dft.py**         # DFT specific settings
├── **dft/**
│   ├── **__init__.py**
│   ├── **runner.py**          # QERunner class
│   └── **error_handler.py**   # Basic error handling (e.g. restart)
├── **utils/**
│   ├── **__init__.py**
│   └── **logging.py**         # Centralised logging
└── **app.py**                 # CLI entry point (basic version)
```

## 3. Design Architecture

### 3.1 Configuration (Pydantic Models)

The system relies on strict configuration schemas to prevent runtime errors.

**`DFTConfig` (in `schemas/dft.py`)**
-   **Responsibilities**: Defines all parameters required to run Quantum Espresso.
-   **Fields**:
    -   `command`: str (e.g., "mpirun -np 4 pw.x"). Validated to ensure no shell injection characters.
    -   `pseudopotential_dir`: Path. Must exist.
    -   `pseudopotentials`: Dict[str, str]. Mapping of element to filename (e.g., {"Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF"}).
    -   `kspacing`: float. Inverse distance for K-point grid generation (default 0.05).
    -   `scf_params`: Dict. Overrides for QE input (mixing_beta, etc.).

**`Element` (in `schemas/common.py`)**
-   **Responsibilities**: Validates chemical symbols.

### 3.2 The Oracle (`QERunner`)

**`QERunner` (in `dft/runner.py`)**
-   **Responsibilities**:
    -   Convert `ase.Atoms` objects into Quantum Espresso input files.
    -   Execute the calculation in a subprocess (securely).
    -   Parse the output to retrieve Energy, Forces, and Stress.
    -   Handle "crashes" gracefully.
-   **Key Methods**:
    -   `compute(atoms: Atoms) -> DFTResult`: Main entry point.
    -   `_write_input(atoms: Atoms, path: Path)`: Internal helper using ASE.
    -   `_run_command(cwd: Path)`: Executes the binary.

**`DFTResult` (Data Model)**
-   **Fields**:
    -   `energy`: float (eV)
    -   `forces`: Array (eV/A)
    -   `stress`: Array (Voigt or full tensor)
    -   `converged`: bool

## 4. Implementation Approach

1.  **Project Initialization**:
    -   Ensure `pyproject.toml` is correctly configured with `uv`.
    -   Set up `src/mlip_autopipec` package structure.

2.  **Configuration Module**:
    -   Implement `schemas/common.py` and `schemas/dft.py`.
    -   Implement `loader.py` using `pydantic` and `pyyaml`.

3.  **DFT Runner Implementation**:
    -   Create `QERunner`.
    -   Integrate `ase.io.write` for input generation.
    -   Implement `subprocess.run` with `shell=False` for security.
    -   Implement basic output parsing (or use `ase.io.read` if reliable, but custom parsing is often needed for specific error codes).

4.  **CLI Entry Point**:
    -   Create a minimal `app.py` using `typer`.
    -   Add a command `test-dft` that reads a config and runs a calculation on a dummy structure.

## 5. Test Strategy

### 5.1 Unit Testing
-   **Config Validation**: Create valid and invalid YAML files. Assert that `DFTConfig` raises `ValidationError` for missing fields or invalid paths.
-   **Input Generation**: Pass a known `ase.Atoms` object to `QERunner._write_input`. Compare the generated text file against a "golden" reference file to ensure flags like `tprnfor=.true.` are present.

### 5.2 Integration Testing
-   **Mock Execution**: Since running actual DFT is slow and requires binaries, we will Mock `subprocess.run`.
    -   Create a test that simulates `pw.x` returning a successful output string.
    -   Verify `QERunner` parses this string and returns the correct Energy/Forces.
-   **Error Handling**: Simulate `subprocess.run` returning a non-zero exit code. Verify `QERunner` raises a specific exception or returns a result with `converged=False`.
