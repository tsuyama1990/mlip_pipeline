# Cycle 03: Oracle & DFT Automation

## 1. Summary

Cycle 03 addresses the "Oracle" component, which provides the ground-truth data (labels) for the machine learning model. This module interfaces with first-principles Density Functional Theory (DFT) codes to calculate the potential energy, atomic forces, and stress tensor for candidate structures.

We will focus on implementing a robust `DFTOracle` that uses Quantum Espresso (via ASE's `Espresso` calculator). A key feature is the "Self-Healing" mechanism: DFT calculations often fail due to electronic convergence issues (SCF not converging). The Oracle must automatically detect these failures and retry with adjusted parameters (e.g., lower mixing beta, higher smearing temperature) without user intervention.

Additionally, we will implement "Periodic Embedding," a crucial technique for handling local environments extracted from large-scale simulations. This involves cutting out a cluster of atoms from a halted MD snapshot and wrapping it in a periodic box suitable for DFT.

## 2. System Architecture

The following file structure will be modified/created. Files in **bold** are the primary deliverables for this cycle.

```
.
├── config.yaml
├── src/
│   └── mlip_autopipec/
│       ├── core/
│       │   ├── config.py         # Update OracleConfig
│       ├── components/
│       │   ├── base.py
│       │   └── **oracle.py**     # DFT Oracle Implementation
│       └── utils/
│           ├── **dft_utils.py**  # Helper functions (embedding, kpoints)
│           └── **constants.py**  # Default DFT parameters
└── tests/
    ├── **test_oracle.py**
    └── **test_dft_utils.py**
```

## 3. Design Architecture

### DFT Oracle (`components/oracle.py`)
The `DFTOracle` implements the `BaseOracle` interface. It manages a pool of worker processes (using `multiprocessing` or `concurrent.futures`) to run DFT calculations in parallel.

*   `compute(structures) -> List[Structure]`:
    1.  For each structure, instantiate an ASE Calculator (e.g., `Espresso`).
    2.  Set up the calculation directory.
    3.  Run the calculation (`atoms.get_potential_energy()`).
    4.  If successful, extract results and return.
    5.  If failed (`ase.calculators.calculator.CalculationFailed`), trigger `_heal_and_retry()`.

### Self-Healing Logic
Private method `_heal_and_retry(atoms, error, attempt)`:
*   **Strategy 1 (Mixing)**: Reduce `mixing_beta` (e.g., 0.7 -> 0.3).
*   **Strategy 2 (Smearing)**: Increase `smearing` width (e.g., 0.01 -> 0.02 Ry).
*   **Strategy 3 (Algorithm)**: Change diagonalization algorithm (e.g., `david` -> `cg`).

### DFT Utilities (`utils/dft_utils.py`)
*   `generate_kpoints(atoms, density)`: Calculates the optimal k-point mesh based on cell dimensions and a target density (e.g., 0.03 A^-1).
*   `create_embedded_cluster(atoms, center_index, radius, buffer)`: Extracts a local environment and creates a periodic supercell.
    *   **Logic**:
        1.  Select atoms within `radius`.
        2.  Select buffer atoms within `radius + buffer`.
        3.  Create a box that bounds these atoms + vacuum (if surface) or periodic (if bulk).
        4.  Return the new `Atoms` object.

## 4. Implementation Approach

1.  **DFT Utilities**: Implement `utils/dft_utils.py`. Ensure k-point generation is robust for non-orthogonal cells.
2.  **Oracle Implementation**: Implement `components/oracle.py`. Start with a single-threaded version.
3.  **ASE Integration**: Use `ase.calculators.espresso.Espresso`. Ensure `tprnfor=True` and `tstress=True` are set.
4.  **Self-Healing**: Implement the retry loop with specific error catching.
5.  **Parallelization**: Add `ProcessPoolExecutor` to handle batch processing.
6.  **Configuration**: Update `OracleConfig` to include paths to pseudopotentials and the `pw.x` executable.

## 5. Test Strategy

### Unit Testing
*   **K-Points**: Create various cells (cubic, hexagonal) and assert `generate_kpoints` returns reasonable grids (e.g., larger cell -> fewer k-points).
*   **Embedding**: Create a large supercell with a known defect. Extract the cluster around the defect. Verify the number of atoms and the cell dimensions.

### Integration Testing (with Mocks/Stubs)
*   **Self-Healing Test**:
    *   Mock the `Espresso` calculator to raise `CalculationFailed` on the first call, then succeed on the second call *only if* `mixing_beta` is < 0.5.
    *   Run `DFTOracle.compute()`.
    *   Assert that the oracle retries and returns the result.
    *   Assert that the log shows "Retrying calculation with mixing_beta=0.3".

### Integration Testing (Real DFT - Optional)
*   **Small System**: If `pw.x` is available, run a calculation on a single water molecule or a 2-atom Si cell.
*   **Verification**: Check that output energy is negative and forces are nearly zero (if relaxed) or non-zero (if distorted).
