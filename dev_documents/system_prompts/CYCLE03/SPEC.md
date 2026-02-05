# CYCLE 03 Specification: Oracle Module Integration

## 1. Summary

In this cycle, we replace the `MockOracle` with a real **`EspressoOracle`**. This module uses the Atomic Simulation Environment (ASE) to interface with **Quantum Espresso (QE)** (or other DFT codes). Key features include "Self-Healing" (automatic error recovery) and "Periodic Embedding" to handle cluster calculations in a periodic code.

## 2. System Architecture

Files to be modified/created:

```ascii
src/mlip_autopipec/
├── config/
│   └── **dft_config.py**           # DFT/QE settings
├── oracle/
│   ├── **__init__.py**
│   ├── **espresso.py**             # QE Wrapper with Self-Healing
│   └── **embedding.py**            # Periodic Embedding Logic
└── orchestration/
    └── orchestrator.py             # Update to use EspressoOracle
```

## 3. Design Architecture

### 3.1. `DFTConfig` (Pydantic)
*   Parameters:
    *   `command`: str (e.g., "pw.x")
    *   `pseudopotentials`: dict[str, str] (path to UPF files)
    *   `kspacing`: float (default 0.04)
    *   `smearing`: str (e.g., "mv", width=0.01)
    *   `self_healing`: bool (default True)

### 3.2. `EspressoOracle` Class
*   **Interface**: Implements `Oracle` protocol.
*   **Method**: `calculate(structures) -> list[StructureMetadata]`
*   **Self-Healing Logic**:
    *   Wrap ASE's `get_potential_energy()` in a `try-except` block.
    *   Catch specific QE errors (e.g., "convergence not achieved").
    *   **Retry Strategy**:
        1.  Reduce `mixing_beta` (0.7 -> 0.3).
        2.  Increase `smearing` temperature.
        3.  Switch diagonalization algorithm (`david` -> `cg`).

### 3.3. `PeriodicEmbedding` (Utility)
*   **Problem**: MD often generates isolated clusters or surface slabs that need to be calculated in a periodic code like QE.
*   **Solution**:
    *   If the structure is non-periodic (cluster), place it in a large box (vacuum buffering).
    *   If it is a local region extracted from MD, ensure the box dimensions respect the cut-off radius to avoid self-interaction artifacts.

## 4. Implementation Approach

1.  **ASE Setup**: Ensure `ase` is installed.
2.  **Config**: Add `DFTConfig`.
3.  **Embedding Logic**: Implement `src/mlip_autopipec/oracle/embedding.py` to handle `atoms.center(vacuum=10.0)`.
4.  **Oracle Logic**:
    *   Create `src/mlip_autopipec/oracle/espresso.py`.
    *   Implement `_run_calc(atoms, params)` method.
    *   Implement the retry loop for self-healing.
5.  **Environment Variables**: The `command` path (e.g., `mpirun -np 4 pw.x`) should be configurable via env vars or `config.yaml`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_embedding.py`**: Verify that a small molecule is correctly centered in a sufficiently large unit cell.
*   **`test_self_healing_logic.py`**:
    *   Mock the ASE Calculator to raise an error on the first call and succeed on the second.
    *   Verify that `EspressoOracle` changes the parameters and retries.

### 5.2. Integration Testing
*   **`test_dft_execution.py`**:
    *   **Mock Mode**: Mock `ase.io.write` (input generation) and `subprocess.run` (QE execution). Verify that the correct input file content is written.
    *   **Real Mode** (if `pw.x` exists): Run a tiny static calculation (e.g., H2 molecule). Check if energy is reasonable (< 0).
