# Cycle 03 Specification: Oracle (DFT Automation)

## 1. Summary

In this cycle, we replace the `MockOracle` with a functional `DFTOracle` capable of running Quantum Espresso (QE) calculations. The Oracle is responsible for generating ground-truth data (energy, forces, stresses) for the structures proposed by the Generator. A critical feature is the **Self-Healing Mechanism**, which automatically detects convergence failures and adjusts calculation parameters (e.g., mixing beta, smearing) to recover without user intervention. Additionally, we implement **Periodic Embedding** to correctly handle cluster calculations within a periodic boundary condition.

## 2. System Architecture

The following file structure will be created. **Bold** files are the targets for this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── oracle/
│   │   ├── **qe.py**               # Quantum Espresso Interface
│   │   └── **base.py**             # (Update ABC with error handling)
│   └── **embedding.py**            # Periodic Embedding Logic
└── utils/
    └── **dft.py**                  # DFT Input Generation Helpers
tests/
└── **test_oracle.py**              # Tests for DFT Execution & Healing
```

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/config.py`)
*   **`DFTConfig`**:
    *   `command`: str (e.g., `mpirun -np 4 pw.x`)
    *   `pseudopotentials`: Dict[str, str] (Path to UPF files)
    *   `kspacing`: float (Target k-point density, e.g., 0.04)
    *   `scf_params`: Dict (Default `conv_thr`, `mixing_beta`)

### 3.2. Oracle Components (`components/oracle/qe.py`)
*   **`DFTOracle`**:
    *   `compute(structures) -> List[Structure]`: The main entry point.
    *   `_run_single_calculation(structure)`: Orchestrates the calculation for one structure.
        1.  **Generate Input**: Create `pw.in` using `ase.io.write`. Ensure `tprnfor=.true.` and `tstress=.true.`.
        2.  **Execute**: Run the `command` using `subprocess`.
        3.  **Parse Output**: Read `pw.out`. If successful, return `Structure` with `energy`, `forces`, `stress`.
        4.  **Handle Error**: If SCF fails, catch exception.
            *   **Retry 1**: Reduce `mixing_beta` (0.7 -> 0.3).
            *   **Retry 2**: Increase `smearing` (0.01 -> 0.05).
            *   **Retry 3**: Switch algorithm (`david` -> `cg`).
            *   **Give Up**: Log warning and discard structure.

### 3.3. Embedding Logic (`components/embedding.py`)
*   **`PeriodicEmbedding`**:
    *   `embed(cluster: Atoms, vacuum: float) -> Atoms`:
        *   Takes a non-periodic cluster (or extracted region).
        *   Places it in a sufficiently large box (vacuum padding).
        *   Ensures the box is orthogonal if required by DFT code.
        *   (Advanced) Supports "Force Masking": Mark atoms in the buffer region to have 0 weight during training.

## 4. Implementation Approach

1.  **DFT Config**: Define `DFTConfig` and update `config.py`.
2.  **Input Generation**: Implement helper functions in `utils/dft.py` to calculate k-points based on `kspacing` and structure size.
3.  **QE Interface**: Implement `DFTOracle` using `ase.calculators.espresso` or raw file I/O.
    *   **Crucial**: Implement the `try...except` block for Self-Healing.
4.  **Embedding**: Implement simple vacuum padding logic in `embedding.py`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_dft_utils.py`**:
    *   Verify `kspacing` logic: Large cell -> fewer k-points; Small cell -> more k-points.
    *   Verify input file generation: Check for `tprnfor=.true.` flag.

### 5.2. Integration Testing (Mocked Binary)
*   **`test_oracle.py`**:
    *   **Mock `subprocess.run`**: Simulate a QE failure (return non-zero exit code or specific error string in stdout).
    *   **Verify Healing**: Assert that `DFTOracle` retries with updated parameters (e.g., check that the second `pw.in` has `mixing_beta = 0.3`).
    *   **Verify Success**: Assert that valid output is parsed correctly into a `Structure` object.

### 5.3. External Integration (Optional/Local)
*   If `pw.x` is available, run a tiny calculation (H2 molecule) to verify the full pipeline.
