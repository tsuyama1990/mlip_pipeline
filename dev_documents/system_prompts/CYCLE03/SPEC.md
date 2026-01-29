# Cycle 03 Specification: DFT Oracle

## 1. Summary

Cycle 03 implements the **Oracle** module, which serves as the interface to the Quantum Espresso (QE) DFT engine. This module is critical for generating ground-truth data (energy, forces, stress) used to train the potential. The key features are automated input file generation (using standard pseudopotentials), robust execution handling, and a self-correction mechanism that attempts to fix common SCF convergence errors by adjusting parameters like mixing beta or smearing.

## 2. System Architecture

### File Structure

Files to be created/modified are **bold**.

```ascii
src/mlip_autopipec/
├── domain_models/
│   ├── config.py                     # Update: Add DFTConfig
│   └── **dft.py**                    # DFTResult models
├── modules/
│   └── **oracle/**
│       ├── **__init__.py**
│       ├── **runner.py**             # Main Oracle class
│       ├── **qe_handler.py**         # QE specific logic (ASE wrapper)
│       └── **error_handler.py**      # Self-correction logic
└── orchestration/
    └── phases/
        ├── **__init__.py**
        └── **oracle.py**             # OraclePhase implementation
```

## 3. Design Architecture

### Domain Models

#### `config.py`
*   **`DFTConfig`**:
    *   `code`: Enum (QE, VASP) - currently only QE.
    *   `command`: str (e.g., "mpirun -np 4 pw.x")
    *   `pseudopotentials`: Dict[str, str] (path or SSSP identifier)
    *   `kspacing`: float (target K-space density)
    *   `scf_params`: Dict (ecutwfc, conv_thr, etc.)

#### `dft.py`
*   **`DFTResult`**:
    *   `structure`: Structure (final)
    *   `energy`: float
    *   `forces`: Array[N, 3]
    *   `stress`: Array[3, 3]
    *   `converged`: bool
    *   `meta`: Dict (computation time, parameters used)

### Components (`modules/oracle/`)

#### `qe_handler.py`
*   **`QERunner`**:
    *   Uses `ase.calculators.espresso.Espresso`.
    *   **`prepare_input(structure, params)`**: Generates `input.pwi`.
    *   **`run(structure) -> DFTResult`**: Executes the calculation.
    *   **`_get_kpoints(structure, kspacing)`**: Calculates K-grid dynamically.

#### `error_handler.py`
*   **`ErrorCorrector`**:
    *   Parses QE output/error logs.
    *   **`diagnose(log_content) -> ErrorType`**.
    *   **`propose_fix(params, error_type) -> NewParams`**: E.g., if "convergence not achieved", reduce `mixing_beta`.

### Orchestration (`orchestration/phases/oracle.py`)

#### `OraclePhase`
*   Iterates over `state.candidates`.
*   Filters out candidates that are already computed.
*   Calls `QERunner.run` for each.
*   Handles exceptions and saves successful `DFTResult`s to `state.results`.

## 4. Implementation Approach

1.  **Update Config**: Add `DFTConfig`.
2.  **Implement Domain Model**: Create `dft.py`.
3.  **Implement QERunner**:
    *   Wrap ASE Espresso calculator.
    *   Ensure `tprnfor=True` and `tstress=True` are set.
4.  **Implement Error Handler**:
    *   Create a simple state machine: Try Default -> Fail -> Try Fix 1 -> Fail -> Try Fix 2 -> Give Up.
5.  **Implement Oracle Phase**:
    *   Integrate into the main loop.
    *   Ensure results are serialized to disk immediately (don't lose expensive data).

## 5. Test Strategy

### Unit Testing
*   **`test_qe_handler.py`**:
    *   Test K-point generation logic (smaller cell -> more K-points).
    *   Mock `subprocess.run` to simulate QE execution and output parsing.
*   **`test_error_handler.py`**:
    *   Feed fake error logs (e.g., "convergence not achieved") and verify the proposed fix (reduced beta).

### Integration Testing
*   **`test_oracle_phase.py`**:
    *   Mock the actual QE binary with a shell script that just writes an output file.
    *   Run the phase and check if `DFTResult` objects are created correctly.
