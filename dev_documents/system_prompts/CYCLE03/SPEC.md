# Cycle 03 Specification: Oracle (DFT Automation)

## 1. Summary

Cycle 03 focuses on the "Teacher" component: the Oracle. This module is responsible for performing First-Principles (DFT) calculations to label the structures generated in Cycle 02. The reliability of the MLIP depends entirely on the quality of this data.

Key features to implement:
1.  **DFT Code Support**: Primarily Quantum Espresso (QE), with architectural support for VASP.
2.  **Self-Correction**: DFT calculations often fail (SCF non-convergence). The Oracle must detect these failures and autonomously retry with robust parameters (e.g., increased smearing, reduced mixing beta).
3.  **Periodic Embedding**: When the Dynamics Engine (Cycle 06) returns a non-periodic cluster (e.g., a local region around a defect), the Oracle must "embed" it into a periodic supercell suitable for plane-wave DFT codes.
4.  **Resource Management**: Efficiently managing `mpirun` execution and core allocation.

By the end of this cycle, the system will be able to take a list of raw `Structure` objects, compute their Energy, Forces, and Virial Stress using DFT, and return labeled `Structure` objects.

## 2. System Architecture

This cycle focuses on the `components/oracle` package.

### File Structure
Files to be created/modified in this cycle are marked in **bold**.

```
src/mlip_autopipec/
├── components/
│   ├── oracle/
│   │   ├── __init__.py
│   │   ├── **base.py**             # Enhanced Abstract Base Class
│   │   ├── **qe_driver.py**        # Quantum Espresso implementation
│   │   ├── **vasp_driver.py**      # VASP implementation (Skeleton)
│   │   ├── **embedder.py**         # Periodic Embedding Logic
│   │   └── **monitor.py**          # Process monitoring & Error detection
│   └── factory.py                  # Update to register Oracles
├── domain_models/
│   └── **config.py**               # Add OracleConfig details
└── tests/
    └── **test_oracle.py**
```

## 3. Design Architecture

### 3.1. Oracle Configuration (`domain_models/config.py`)
Update `OracleConfig` to include:
*   `type`: "qe" or "vasp".
*   `command`: String (e.g., "mpirun -np 4 pw.x").
*   `pseudo_dir`: Path to pseudopotentials.
*   `kspacing`: Float (Target density in inverse Angstrom, e.g., 0.04).
*   `scf_params`: Dict (Defaults for mixing, smearing).

### 3.2. DFT Driver Logic (`components/oracle/qe_driver.py`)
We will use `ase.calculators.espresso.Espresso` but wrap it to handle the complex I/O and error recovery.
*   **Input**: `Structure` (ASE Atoms).
*   **Output**: `Structure` with `labels` updated.
*   **Self-Correction Loop**:
    1.  Attempt calculation with standard params.
    2.  Catch `ase.calculators.calculator.CalculationFailed`.
    3.  Parse error (e.g., "convergence not achieved").
    4.  Modify params (mixing 0.7 -> 0.3).
    5.  Retry (Max 3 attempts).

### 3.3. Periodic Embedder (`components/oracle/embedder.py`)
*   **Problem**: MD halts on a local cluster. DFT needs periodic boundary conditions (PBC).
*   **Solution**:
    1.  Identify the "active region" (the atoms passed from MD).
    2.  Place them in a large box (Vacuum padding).
    3.  (Advanced) If the cluster represents a bulk defect, reconstruct the bulk periodicity around it.
    4.  For Cycle 03, we implement the "Box Padding" strategy: Ensure at least 10Å vacuum for non-periodic clusters to avoid spurious interactions.

## 4. Implementation Approach

1.  **Implement `PeriodicEmbedder`**: Create a utility class that takes an `Atoms` object and ensures it has a cell and PBCs suitable for DFT.
2.  **Implement `QEOracle`**:
    *   Initialize with `OracleConfig`.
    *   Implement `compute(structures)`:
        *   Iterate through structures.
        *   Apply embedding if needed.
        *   Write input file (`pw.x < in > out`).
        *   Run subprocess.
        *   Read output using ASE.
        *   Handle exceptions.
3.  **Mocking for Development**: Since we might not have `pw.x` in the CI environment, create a `MockOracle` that returns Lennard-Jones energies but simulates the "delay" and "interface" of a real DFT code.
4.  **Orchestrator Integration**: Add the Oracle step after Generation.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Embedder**:
    *   Input: A cluster of 4 atoms with no cell.
    *   Action: `embed(atoms)`.
    *   Assert: Atoms have a cell (e.g., 20x20x20) and `pbc=[True, True, True]`.
*   **QE Driver (Mocked)**:
    *   Input: `atoms` object.
    *   Mock: `subprocess.run` returns a string containing "Total Energy = -123.4 Ry".
    *   Action: `oracle.compute([atoms])`.
    *   Assert: Returned atoms have `energy` attribute set.

### 5.2. Integration Testing
*   **Self-Correction Test**:
    *   Mock `subprocess.run` to fail once (return error code) then succeed on second call.
    *   Assert that `oracle.compute()` logs a warning "Retrying with reduced mixing..." and eventually returns success.
