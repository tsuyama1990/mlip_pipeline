# Cycle 02 Specification: Oracle (DFT Automation)

## 1. Summary

In Cycle 02, we implement the "Oracle" module, which is responsible for performing Density Functional Theory (DFT) calculations. This module acts as the ground truth generator for the machine learning potential. The primary challenge here is not just running the calculation, but doing so robustly and efficiently. DFT calculations often fail due to electronic convergence issues, especially for the high-energy, distorted structures generated during active learning.

We will implement a `QERunner` (Quantum Espresso Runner) that features a "Self-Healing" mechanism. If a calculation fails, the runner will automatically adjust parameters (e.g., mixing beta, smearing width) and retry, significantly reducing manual intervention. Additionally, we will implement the "Periodic Embedding" technique. This allows us to calculate forces for a specific local environment by cutting out a small supercell, wrapping it in periodic boundary conditions, and running DFT on this reduced system, rather than the entire large MD simulation box. This is crucial for computational efficiency.

## 2. System Architecture

**Files to be created/modified in this cycle are marked in Bold.**

```ascii
src/mlip_autopipec/
├── dft/
│   ├── **__init__.py**
│   ├── **runner.py**           # QERunner class
│   ├── **inputs.py**           # Input file generation logic
│   ├── **embedding.py**        # Periodic Embedding logic
│   └── **errors.py**           # DFT-specific exception handlers
└── config/
    └── dft.py                  # (Modified to add more tuning params)
```

## 3. Design Architecture

### `QERunner` Class (in `dft/runner.py`)
*   **Responsibility**: Execute the `pw.x` command via `subprocess`.
*   **Key Methods**:
    *   `run_single_point(atoms: Atoms, run_dir: Path) -> DFTResult`: Runs a static SCF calculation.
    *   `_execute_with_retry(command, cwd)`: Internal method handling the self-healing loop.
*   **Self-Healing Logic**:
    *   We define a hierarchy of "Recovery Strategies".
    *   Strategy 1: Default parameters.
    *   Strategy 2: Reduce `mixing_beta` (e.g., 0.7 -> 0.3).
    *   Strategy 3: Increase `electron_maxstep` (e.g., 100 -> 200).
    *   Strategy 4: Increase `smearing` (electronic temperature).
    *   If all strategies fail, raise `DFTConvergenceError`.

### `InputGenerator` Class (in `dft/inputs.py`)
*   **Responsibility**: Convert an ASE `Atoms` object into a Quantum Espresso input string.
*   **Features**:
    *   **SSSP Integration**: Automatically map element names to pseudopotential filenames (config-defined).
    *   **K-Spacing**: Calculate `k_points` grid dimensions dynamically based on cell size and target `kspacing` ($N_i = \text{int}(2\pi / (L_i \times \text{kspacing})) + 1$).
    *   **Directives**: Always set `tprnfor=.true.` and `tstress=.true.` to extract forces and stress.

### `PeriodicEmbedding` (in `dft/embedding.py`)
*   **Concept**:
    1.  Select "Active Region" (atoms with high uncertainty).
    2.  Select "Buffer Region" (atoms within $R_{cut} + \Delta$ of Active Region).
    3.  Create a bounding box that encapsulates these atoms.
    4.  **Crucial Step**: Ensure the box dimensions allow for periodic boundary conditions (minimum image convention) if we treat it as a new unit cell.
    5.  Return a new `Atoms` object representing this "embedded" cluster as a periodic system.

## 4. Implementation Approach

1.  **Input Generation**: Start by implementing `InputGenerator`. This is pure logic (Atoms -> String) and easy to test.
2.  **Runner Skeleton**: Implement `QERunner` basic execution.
3.  **Self-Healing**: Add the retry loop. Use a state machine or simple list of configuration dictionaries to iterate through strategies.
4.  **Embedding Logic**: Implement the geometric manipulations in `embedding.py`. Use `ase.neighborlist` or `ase.geometry` functions.
5.  **Integration**: Update `Orchestrator` to instantiate `QERunner` (though not yet used in a loop).

## 5. Test Strategy

### Unit Testing
*   **Input Gen**:
    *   Pass a silicon unit cell. Verify the generated text contains correct `ATOMIC_POSITIONS`, `K_POINTS`, and `CELL_PARAMETERS`.
    *   Verify `k_points` scaling: doubling cell size should halve the grid density.
*   **Embedding**:
    *   Create a large supercell with a known defect.
    *   Run `embed(defect_index)`.
    *   Verify the returned atoms object is smaller than the original but contains the defect and neighbors.
    *   Verify the cell is orthogonal (or as expected).

### Integration Testing
*   **Mocked Execution**:
    *   Since we cannot assume `pw.x` is installed in the CI environment, we must mock `subprocess.run`.
    *   **Test Self-Healing**: Mock the first call to return a "Convergence NOT achieved" string in stdout. Mock the second call (with different params) to return "JOB DONE". Assert that the runner retries and succeeds.
    *   **Output Parsing**: Feed a real `pw.x` output file (saved as fixture) to the parser and verify it extracts the correct Energy, Forces, and Stress.
