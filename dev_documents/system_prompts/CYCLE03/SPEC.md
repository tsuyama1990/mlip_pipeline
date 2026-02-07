# Cycle 03 Specification: Oracle & Trainer Integration

## 1. Summary

**Goal**: Enable real physics capabilities by implementing the interfaces to **Quantum Espresso (QE)** and **Pacemaker**. This cycle replaces the mocks from Cycle 01 with actual computational engines. The `DFTManager` will handle self-healing DFT calculations, and the `PacemakerWrapper` will execute the `pace_train` command to fit ACE potentials.

**Key Deliverables**:
1.  **`DFTManager` (Oracle)**: A robust wrapper around ASE's `Espresso` calculator. It must handle input file generation, `pw.x` execution, and crucially, automatic error recovery (e.g., reducing mixing beta if SCF fails).
2.  **`PacemakerWrapper` (Trainer)**: A wrapper around the `pace_train` CLI tool. It manages the training process, including active set selection and delta learning configuration.
3.  **Configuration**: Comprehensive `OracleConfig` and `TrainerConfig` models to support these tools.

## 2. System Architecture

Files in **bold** are the primary focus of this cycle.

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── **config.py**         # Enhanced OracleConfig (QE settings)
├── infrastructure/
│   ├── oracle/
│   │   ├── **__init__.py**
│   │   └── **qe.py**         # DFTManager (ASE/Espresso wrapper)
│   └── trainer/
│       ├── **__init__.py**
│       └── **pacemaker.py**  # PacemakerWrapper (pace_train)
└── utils/
    └── **process.py**        # Subprocess management
```

## 3. Design Architecture

### 3.1 `DFTManager` (Oracle)

*   **Config**: `command` (str), `pseudopotentials` (Dict), `kspacing` (float), `smearing` (float).
*   **Logic**:
    1.  Convert `Structure` to `ase.Atoms`.
    2.  Set up `Espresso` calculator with parameters.
    3.  Execute `atoms.get_potential_energy()`.
    4.  **Self-Healing**: Catch `ase.calculators.calculator.PropertyNotImplementedError` or specific QE errors.
        *   Retry 1: Reduce `mixing_beta` (0.7 -> 0.3).
        *   Retry 2: Increase `smearing`.
        *   Retry 3: Change diagonalisation method (`david` -> `cg`).
    5.  Return labeled `Structure` with `energy`, `forces`, `stress`.

### 3.2 `PacemakerWrapper` (Trainer)

*   **Config**: `command` (str), `ladder_step` (List), `kappa` (float), `max_basis_sise` (int).
*   **Logic**:
    1.  Convert `Dataset` (list of structures) to Pacemaker's expected format (Pickle/ExtXYZ).
    2.  Construct `pace_train` command line arguments.
    3.  Execute `subprocess.run()`.
    4.  Parse the output log to extract metrics (RMSE).
    5.  Return a `Potential` object pointing to the generated `.yace` file.

## 4. Implementation Approach

1.  **Enhance Config**: Update `domain_models/config.py` with specific fields for QE and Pacemaker.
2.  **Implement `DFTManager`**: Use `ase.calculators.espresso`. Mock `subprocess.run` in tests to simulate `pw.x` output.
3.  **Implement `PacemakerWrapper`**: Use `subprocess.run` to call `pace_train`.
4.  **Integration Test**: Verify that `Orchestrator` can switch from `mock` to `qe`/`pacemaker` via config.

## 5. Test Strategy

### 5.1 Unit Testing
*   **DFT Input Generation**: Verify that `DFTManager` generates correct `input_data` dictionary for ASE.
*   **Self-Healing Logic**: Mock a failed calculation (raise Exception) and verify that `get_potential_energy` is called again with different parameters.
*   **Pacemaker Command**: Verify the constructed command string is correct.

### 5.2 Integration Testing
*   **"Mock Binary" Test**: Create dummy scripts (`mock_pw.x`, `mock_pace_train`) that simulate the behaviour of the real binaries (e.g., write an output file and exit 0). Configure the wrappers to use these mocks.
*   **Real Execution (Optional/Local)**: If `pw.x` is available, run a tiny calculation (H2 molecule) to verify end-to-end functionality.
