# Cycle 03 Specification: Oracle (DFT Automation)

## 1. Summary

In Cycle 03, we implement the **Oracle**, the component responsible for generating ground-truth data using Density Functional Theory (DFT). We focus on integrating with **Quantum Espresso (QE)** via the ASE interface. The critical challenge here is robustness: DFT calculations often fail due to electronic convergence issues. Therefore, we implement an **Self-Healing Mechanism** that automatically detects failures and retries with adjusted parameters (e.g., increased smearing, reduced mixing beta). We also implement **Periodic Embedding**, a technique to carve out small, calculable clusters from large MD snapshots while preserving the correct boundary conditions for the forces.

## 2. System Architecture

Files in **bold** are the focus of this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── oracle/
│   │   ├── __init__.py
│   │   ├── **espresso.py**     # Quantum Espresso implementation
│   │   ├── **vasp.py**         # (Placeholder for future)
│   │   ├── **embedding.py**    # Periodic Embedding logic
│   │   └── **scheduler.py**    # Job management (local/slurm)
├── domain_models/
│   ├── **config.py**           # Add QeConfig
```

## 3. Design Architecture

### 3.1. EspressoOracle (`components/oracle/espresso.py`)
Inherits from `BaseOracle`.
-   **Config**: `command` (e.g., `mpirun -np 4 pw.x`), `pseudopotentials` (dict), `kspacing` (float).
-   **Method `compute(structures)`**:
    -   Iterates through structures.
    -   Applies `PeriodicEmbedding` if the structure is a cluster cut from MD.
    -   Sets up `ase.calculators.espresso.Espresso`.
    -   Runs calculation.
    -   Extracts `energy`, `forces`, `stress`.
    -   **Error Handling**: Wraps the call in a `SelfCorrection` loop.

### 3.2. Self-Correction Strategy
If `calc.get_potential_energy()` raises an `EspressoError` (SCF non-convergence):
1.  **Attempt 1**: Reduce `mixing_beta` (0.7 -> 0.3).
2.  **Attempt 2**: Increase `smearing` (0.01 -> 0.05 Ry).
3.  **Attempt 3**: Change mixing mode (plain -> local-TF).
4.  **Fail**: Mark structure as "Failed" and skip, logging the error.

### 3.3. Periodic Embedding (`components/oracle/embedding.py`)
-   **Input**: Large MD Supercell + Center Atom Index + Radius.
-   **Output**: Small Orthorhombic Cell.
-   **Logic**:
    1.  Select atoms within `radius` + `buffer`.
    2.  Construct a minimal bounding box.
    3.  Enforce periodic boundary conditions on this box (approximate).
    4.  Mask forces on buffer atoms (weight=0 in training) so only the center is learned.

## 4. Implementation Approach

1.  **Update Config**: Add `QeConfig` to `domain_models/config.py`.
2.  **Implement Embedding**: Create `embedding.py`. Use `ase.geometry.get_distances`.
3.  **Implement EspressoOracle**:
    -   Use `subprocess` or `ase`'s internal runner.
    -   Implement the retry logic using a `RetryingCalculator` wrapper.
4.  **Dependency**: Ensure `ase` is installed.
5.  **Test**: Since we don't have QE installed in CI, we will create a "MockEspresso" for tests that mimics the input/output files of QE without running the binary.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Embedding**: Create a 1000-atom cell. Cut a 5Å cluster. Verify the number of atoms is small (~50) and pbc is True.
-   **Self-Correction**:
    -   Mock `calc.get_potential_energy` to raise an exception on the first call and succeed on the second.
    -   Verify that the Oracle catches the error, updates parameters, and retries.

### 5.2. Integration Testing
-   **Scenario**: "Mock QE Flow"
-   **Config**: Oracle=`espresso`, command=`echo "JOB DONE" > espresso.out`.
-   **Note**: Real integration tests require the `pw.x` binary. We will skip execution if binary is missing, or use a "Dry Run" mode where ASE generates the `espresso.pwi` file and we verify its content (checking k-points, pseudos).
