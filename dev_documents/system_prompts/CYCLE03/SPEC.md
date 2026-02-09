# Cycle 03 Specification: Oracle (DFT)

## 1. Summary
The "Oracle" is the source of ground truth for the active learning pipeline. It is responsible for taking candidate structures from the Generator or Dynamics Engine and performing high-fidelity Density Functional Theory (DFT) calculations to obtain energy, forces, and stresses. This cycle focuses on automating the Quantum Espresso (QE) workflow, including robust error handling ("Self-Healing") and efficient data extraction via "Periodic Embedding". By the end of this cycle, the system will be able to autonomously run SCF calculations, recover from convergence failures, and produce high-quality training data.

## 2. System Architecture

Files in **bold** are to be created or modified.

```ascii
src/mlip_autopipec/
├── components/
│   ├── oracle/
│   │   ├── **__init__.py**
│   │   ├── **base.py**             # BaseOracle (ABC)
│   │   ├── **qe_driver.py**        # Quantum Espresso Interface
│   │   ├── **self_healer.py**      # Error recovery logic
│   │   └── **embedding.py**        # Periodic Embedding Logic
│   └── ...
└── core/
    └── **orchestrator.py**         # Update to use oracle
```

## 3. Design Architecture

### 3.1 Quantum Espresso Driver (`qe_driver.py`)

This class wraps `ase.calculators.espresso.Espresso`. It manages:
*   **Input File Generation**: Setting `tprnfor=True`, `tstress=True` (critical for MLIP training).
*   **K-Points**: Using `kspacing` (e.g., 0.03 $\AA^{-1}$) to automatically determine grid density based on cell size.
*   **Pseudopotentials**: Automatically loading SSSP pseudopotentials from a configured directory.
*   **Execution**: Running the `pw.x` command via ASE or subprocess.

### 3.2 Self-Healing Oracle (`self_healer.py`)

A decorator or wrapper around the `compute()` method. It catches `EspressoError` (or generic calculation failures) and retries with modified parameters.

**Retry Strategy:**
1.  **Standard Run**: Default mixing beta (0.7), default smearing.
2.  **Retry 1 (Convergence)**: Reduce mixing beta to 0.3. Increase electron steps.
3.  **Retry 2 (Robustness)**: Switch diagonalization to `cg` (Conjugate Gradient). Increase smearing temperature.
4.  **Fail**: If all retries fail, mark structure as "FAILED" and skip.

### 3.3 Periodic Embedding (`embedding.py`)

Logic to extract a small, trainable cluster from a large MD snapshot.

**Algorithm:**
1.  **Identify Center**: Select the atom with high uncertainty (or random).
2.  **Cut Cluster**: Extract all atoms within radius $R_{cut} + R_{buffer}$ (e.g., 5.0 + 3.0 Å).
3.  **Wrap**: Enclose the cluster in a minimal orthorhombic box that respects periodic boundary conditions (PBC). This is non-trivial; for non-orthogonal cells, we may need to build a supercell of the cut cluster that approximates a cubic box.
    *   *Simplification for Cycle 03*: Just extract the cluster and place it in a large enough box (vacuum padding) if we are training a cluster model, OR constrain the box to be periodic if the original system was periodic.
    *   *Refined Approach*: Use `ase.build.cut` logic or similar to create a valid periodic supercell that contains the local environment.

## 4. Implementation Approach

1.  **Develop Embedding Logic**: Implement `embedding.py`. Test with simple cubic lattices first.
2.  **Implement QE Driver**: Create `QEOracle` inheriting from `BaseOracle`. Ensure it generates valid `pw.x` input strings.
3.  **Implement Self-Healer**: Create the retry loop logic. Mock the `Espresso` calculator to simulate failures (raise Exception on 1st call, succeed on 2nd).
4.  **Integration**: Connect `Oracle` to `Orchestrator`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Input Generation**: Verify that `tprnfor=True` is always present in the generated input dict/string.
*   **K-Spacing**: Verify that a larger cell gets fewer k-points than a smaller cell for the same `kspacing`.
*   **Embedding**:
    *   Input: 1000-atom MD snapshot.
    *   Output: ~50-atom cluster.
    *   Check: The local environment (neighbors) of the central atom in the cluster matches the original snapshot.

### 5.2 Integration Testing
*   **Mock Self-Healing**:
    *   Configure `QEOracle` with a `MockCalculator` that raises `SCFError` once.
    *   Run `compute()`.
    *   Verify it catches the error, adjusts parameters (check logs), and retries.
    *   Verify it returns a valid `Structure` with energy/forces.
*   **Real Execution (Optional)**:
    *   If `pw.x` is in the path, run a calculation on a standard Si unit cell.
    *   Verify output energy is reasonable (approx -150 to -200 eV for 2 atoms, depending on pseudo).
