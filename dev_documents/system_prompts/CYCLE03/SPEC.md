# CYCLE 03 Specification: Oracle & Data Management

## 1. Summary

Cycle 03 focuses on the "Oracle" component, which provides ground-truth data using Density Functional Theory (DFT). We will implement the `EspressoOracle` to interface with Quantum Espresso (via ASE). A critical requirement is "Self-Healing": DFT calculations often fail due to SCF convergence issues. The Oracle must detect these failures and automatically retry with adjusted parameters (e.g., lower mixing beta, increased smearing) without user intervention. Additionally, we will implement "Periodic Embedding", a technique to cut out clusters from large MD snapshots and wrap them in a valid periodic box for DFT calculation, preventing surface artifacts.

## 2. System Architecture

New files and modifications:

```ascii
src/mlip_autopipec/
├── services/
│   ├── external/
│   │   └── espresso_interface.py   # [CREATE] Wrapper for ASE Espresso calculator
│   └── oracle.py                   # [CREATE] Concrete EspressoOracle with retry logic
└── utils/
    └── embedding.py                # [CREATE] Periodic Embedding logic
```

## 3. Design Architecture

### Oracle Logic (`oracle.py`)
-   **Class**: `EspressoOracle` implements the `Oracle` protocol.
-   **Method**: `calculate(structures) -> labeled_structures`
-   **Self-Healing State Machine**:
    -   Attempt 1: Standard settings.
    -   Catch `JobFailedError`.
    -   Attempt 2: Reduce `mixing_beta` (0.7 -> 0.3).
    -   Attempt 3: Change diagonalization (david -> cg).
    -   Attempt 4: Increase `smearing`.
    -   Final: Raise Error if all fail.

### Periodic Embedding (`embedding.py`)
-   **Problem**: Active learning selects a local cluster (e.g., a dislocation core) from a large MD system.
-   **Solution**:
    1.  Select atoms within $R_{cut} + R_{buffer}$.
    2.  Define an orthorhombic cell that bounds these atoms.
    3.  Add vacuum? NO. The spec requires **Periodic Embedding**.
    4.  Create a small supercell that repeats this local environment. (Note: True periodic embedding of arbitrary clusters is hard; the spec describes cutting a "Orthorhombic Box" and treating it as a small periodic supercell. We will follow the spec: "包含する直方体状のセル...周期境界条件を持つ小さなスーパーセルとして再定義").

## 4. Implementation Approach

1.  **Implement `PeriodicEmbedding`**:
    -   Function `embed_cluster(atoms, center_index, r_cut, r_buffer)`.
    -   Logic: Extract atoms, center them, define a bounding box, set `pbc=True`.

2.  **Implement `EspressoOracle`**:
    -   Use `ase.calculators.espresso.Espresso`.
    -   Mandatory flags: `tprnfor=True`, `tstress=True` (Energy, Forces, Stress).
    -   Pseudopotentials: Load from configuration (SSSP).

3.  **Implement Retry Logic**:
    -   Wrap the ASE `get_potential_energy()` call in a try/except block.
    -   Parse the error message (or just assume SCF failure) and modify the calculator's `parameters` dictionary before retrying.

## 5. Test Strategy

### Unit Testing
-   **Embedding**: Create a synthetic large lattice. Select an atom. Verify that `embed_cluster` returns a smaller `Atoms` object with correct PBC and included atoms.
-   **Retry Logic**: Mock the ASE calculator to raise an exception on the first call and succeed on the second. Verify that the Oracle correctly retries and returns the result.

### Integration Testing
-   **ASE Interface**: Verify that the generated Quantum Espresso input file (text) contains the correct flags (e.g., `mixing_beta`, `tprnfor`). We won't run actual QE.
