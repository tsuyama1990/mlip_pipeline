# Cycle 02 Specification: The Oracle (DFT Automation)

## 1. Summary

Cycle 02 focuses on the "Oracle" component, which serves as the source of truth for the system. In the context of MLIP, the Oracle is responsible for generating training data by running Density Functional Theory (DFT) calculations. This cycle implements the `DFTManager` class, which handles the submission, monitoring, and parsing of Quantum Espresso (QE) jobs. Crucially, it also implements "Self-Healing" capabilities to handle common DFT convergence failures automatically, and "Periodic Embedding" to correctly handle cluster-in-box geometries for training data generation.

## 2. System Architecture

### 2.1 File Structure

**Bold** files are to be created or modified in this cycle.

```text
.
├── src/
│   └── mlip_autopipec/
│       ├── physics/
│       │   ├── oracle/
│       │   │   ├── **__init__.py**
│       │   │   ├── **manager.py**      # Main DFTManager class
│       │   │   ├── **qe_parser.py**    # QE Input/Output handling
│       │   │   └── **embedding.py**    # Periodic Embedding logic
│       │   └── **__init__.py**
│       └── orchestration/
│           ├── **mocks.py**            # Update MockOracle to be smarter
│           └── orchestrator.py
└── tests/
    ├── unit/
    │   ├── **test_dft_manager.py**
    │   ├── **test_qe_parser.py**
    │   └── **test_embedding.py**
    └── integration/
        └── **test_oracle_integration.py**
```

## 3. Design Architecture

### 3.1 `DFTManager` Class
-   **Role**: Coordinates the execution of DFT calculations.
-   **Responsibilities**:
    -   Accept a list of `ase.Atoms` objects.
    -   Apply the "Periodic Embedding" strategy.
    -   Generate input files for `pw.x`.
    -   Execute the calculation (via `subprocess` or `ase.calculators.espresso`).
    -   Parse results (Energy, Forces, Stress).
    -   Implement retry logic for failed jobs.
-   **Config Dependence**: Depends on `DFTConfig` for parameters (pseudopotentials, k-points).

### 3.2 `PeriodicEmbedding` Logic
-   **Problem**: Active Learning extracts local clusters (non-periodic). DFT (pw.x) requires periodic boundary conditions.
-   **Solution**:
    1.  Create a vacuum box around the cluster.
    2.  Ensure the box is large enough ($> 2 \times R_{cut}$) to prevent self-interaction.
    3.  (Optional) Embed the cluster in a larger supercell of the bulk material (if using QM/MM style, though Cycle 02 focuses on simple vacuum padding first).

### 3.3 Error Handling (The "Self-Healing" Mechanism)
We define a hierarchy of exceptions in `domain_models/exceptions.py` (new file):
-   `DFTConvergenceError`: SCF did not converge.
-   `DFTGeometryError`: Atoms too close.
-   **Recovery Strategy**:
    -   If `DFTConvergenceError`: Decrease `mixing_beta`, increase `electron_maxstep`, or switch `mixing_mode`.
    -   Retry up to 3 times before raising a fatal error.

## 4. Implementation Approach

1.  **QE Parser**:
    -   Implement `write_qe_input(atoms, params, filename)`: Robustly generates `pw.x` input.
    -   Implement `read_qe_output(filename)`: Parses the XML or text output for Energy, Forces, Stress.
    -   *Note*: While ASE has `read_espresso`, we may need a custom wrapper to handle specific flags (`tprnfor`, `tstress`) and error logs reliably.

2.  **Periodic Embedding**:
    -   Implement `embed_in_box(atoms, vacuum=10.0)` function.
    -   This takes a cluster and centers it in a sufficiently large orthorhombic cell.

3.  **DFT Manager Implementation**:
    -   Implement `DFTManager.compute_batch(structures)`.
    -   Use `concurrent.futures.ThreadPoolExecutor` (or ProcessPool) if running multiple local calculations (though usually restricted by cores).
    -   Integrate the **Self-Healing Loop**:
        ```python
        for attempt in range(max_retries):
            try:
                run_calculation()
                break
            except DFTConvergenceError:
                adjust_parameters()
        ```

4.  **Mock Oracle Upgrade**:
    -   Update `MockOracle` to read a "pre-calculated" dataset (XYZ file) and return values from nearest neighbors, or use a simple Lennard-Jones potential (`ase.calculators.lj`) to generate "fake" but physically consistent forces. This is crucial for testing the Trainer in Cycle 04.

## 5. Test Strategy

### 5.1 Unit Testing
-   **`test_embedding.py`**:
    -   Input: A simple CO molecule.
    -   Action: call `embed_in_box`.
    -   Assertion: The cell size is correct, atoms are centered, and pbc is `[True, True, True]`.
-   **`test_qe_parser.py`**:
    -   Input: Dummy `pw.x` output string containing "convergence not achieved".
    -   Assertion: `read_qe_output` raises `DFTConvergenceError`.

### 5.2 Integration Testing
-   **`test_oracle_integration.py`**:
    -   **Mock Mode**: Verify `DFTManager` calls `MockCalculator` and returns labelled atoms.
    -   **Real Mode** (if QE installed): Run a calculation on a single water molecule. Check that the returned energy is negative and forces are non-zero.
