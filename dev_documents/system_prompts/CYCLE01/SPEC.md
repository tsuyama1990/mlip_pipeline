# Cycle 01 Specification: Core Framework & Oracle

## 1. Summary
Cycle 01 establishes the foundational infrastructure of PyAcemaker. The primary goals are to define the strict configuration schemas that drive the "Zero-Config" workflow and to implement the **Oracle** module capable of executing Density Functional Theory (DFT) calculations. Robustness is key; the Oracle must handle common DFT convergence errors automatically ("Self-Healing").

## 2. System Architecture

### 2.1. File Structure
The following files are to be created or modified in this cycle.

```text
src/mlip_autopipec/
├── __init__.py
├── app.py                          # [CREATE] Main CLI entry point
├── config/                         # [CREATE] Configuration Module
│   ├── __init__.py
│   ├── base.py                     # [CREATE] Base Pydantic models
│   ├── dft_config.py               # [CREATE] DFT/QE parameters
│   └── workflow_config.py          # [CREATE] Global workflow settings
├── dft/                            # [CREATE] Oracle Module
│   ├── __init__.py
│   ├── runner.py                   # [CREATE] QERunner class
│   └── calculator.py               # [CREATE] ASE Calculator wrapper
└── orchestration/                  # [CREATE]
    ├── __init__.py
    └── manager.py                  # [CREATE] Basic Workflow Manager
```

### 2.2. Component Interaction
- **CLI (`app.py`)**: Reads `config.yaml`, validates it using `WorkflowConfig`, and instantiates the `WorkflowManager`.
- **`WorkflowManager`**: Orchestrates the process. In Cycle 01, it initializes the `QERunner`.
- **`QERunner`**: Takes atomic structures, writes QE input files (handling pseudopotentials, k-points), executes the `pw.x` binary, and parses output. It implements retry logic for failures.

## 3. Design Architecture

### 3.1. Configuration Models (`src/mlip_autopipec/config/`)
We use Pydantic V2 for strict validation.

- **`DFTConfig`**:
    - `command`: str (e.g., "mpirun -np 4 pw.x")
    - `pseudopotentials`: Dict[str, str] (Element -> Filename)
    - `pseudo_dir`: Path
    - `kspacing`: float (Target k-point density, e.g., 0.04)
    - `scf_params`: Dict (mixing_beta, electron_maxstep, etc.)
- **`WorkflowConfig`**:
    - `project_name`: str
    - `work_dir`: Path
    - `dft`: DFTConfig

### 3.2. Oracle / DFT Runner (`src/mlip_autopipec/dft/`)
- **`QERunner` Class**:
    - **Responsibilities**:
        - Convert ASE `Atoms` to QE input (`.pwi`).
        - Run calculation (blocking or async).
        - Parse Energy, Forces, Stress from output (`.pwo`).
        - **Self-Healing**: Catch `ConvergenceError`. If found, reduce `mixing_beta` or increase `smearing` and retry.
    - **Methods**:
        - `run(atoms: Atoms) -> DFTResult`
        - `_generate_input(atoms: Atoms) -> str`
        - `_check_convergence(output: str) -> bool`

## 4. Implementation Approach

1.  **Project Setup**: Initialize the `src` directory and install dependencies (`ase`, `pydantic`).
2.  **Config Implementation**: Define the Pydantic models in `config/`. Ensure validation works for missing fields or invalid paths.
3.  **DFT Runner Implementation**:
    - Implement `QERunner` using `subprocess` or `ase.calculators.espresso`.
    - Note: We prefer a custom wrapper or extending ASE to have fine-grained control over error handling and input generation (e.g., enforcing `tprnfor=.true.`).
4.  **CLI scaffolding**: Create `app.py` using `typer` or `argparse` to load the config and run a dummy DFT task.

## 5. Test Strategy

### 5.1. Unit Testing
- **Config**: Test loading valid and invalid YAMLs. Verify defaults.
- **QERunner**:
    - Mock the `subprocess.run` calls to simulate successful and failed QE runs.
    - Verify that `Self-Healing` logic triggers a retry with modified parameters when a failure is simulated.
    - Test input file generation (check if `tprnfor` is present).

### 5.2. Integration Testing
- **Real Execution**:
    - Requirement: A working `pw.x` or a mock script that behaves like `pw.x`.
    - Test: Run `mlip-auto run` with a config pointing to a simple Silicon structure. Verify it generates a valid output file.
