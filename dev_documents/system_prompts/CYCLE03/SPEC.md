# Cycle 03: Oracle (DFT Integration)

## 1. Summary

Cycle 03 implements the **Oracle**, the component responsible for generating high-fidelity ground truth data via Density Functional Theory (DFT). This is the most computationally expensive part of the workflow, so robustness and efficiency are paramount.

We will integrate **Quantum Espresso (QE)** via the `ase.calculators.espresso` interface. To handle the inevitable convergence failures of DFT, we implement a **Self-Healing Mechanism** that automatically adjusts calculation parameters (mixing, smearing, algorithm) upon failure.

Additionally, to support the active learning of local environments (e.g., defects, surfaces) from large MD snapshots, we implement **Periodic Embedding**. This technique extracts a relevant cluster from a large system and embeds it into a smaller, periodic supercell suitable for DFT, minimizing the computational cost while preserving the local physics.

## 2. System Architecture

Files in **bold** are new or modified.

```ascii
src/mlip_autopipec/
├── components/
│   ├── oracle/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── **qe.py**             # Quantum Espresso Implementation
│   │   ├── **vasp.py**           # VASP Implementation (Optional/Stub)
│   │   ├── **embedding.py**      # Periodic Embedding Logic
│   │   └── **healing.py**        # Error Handling & Retry Logic
│   └── ...
```

## 3. Design Architecture

### 3.1. DFT Manager (`qe.py`)
-   **Class `QECalculator`**:
    -   Wraps `ase.calculators.espresso.Espresso`.
    -   `compute(structure: Structure) -> Structure`
    -   **Inputs**: `pseudopotentials` (dict), `kspacing` (float), `encut` (float).
    -   **Outputs**: Updated `Structure` with `energy`, `forces`, `stress`.

### 3.2. Self-Healing (`healing.py`)
-   **Class `Healer`**:
    -   `heal(failed_calc: Calculator, error: Exception) -> Calculator`
    -   **Strategies**:
        1.  Reduce `mixing_beta` (e.g., 0.7 -> 0.3).
        2.  Increase `smearing` (electronic temperature).
        3.  Change diagonalization algorithm (`david` -> `cg`).
        4.  If all fail, discard the structure (do not label).

### 3.3. Periodic Embedding (`embedding.py`)
-   **Function `embed_cluster(cluster: Atoms, vacuum: float) -> Atoms`**:
    -   Takes a non-periodic cluster (cut from MD).
    -   Places it in a box with sufficient vacuum ($> 2 \times R_{cut}$).
    -   Optionally, if the cluster was from a bulk crystal, embeds it into a perfect bulk supercell (QM/MM style - advanced).
    -   For now, we focus on **Cluster-in-Vacuum** approach with periodic boundary conditions (supercell).

## 4. Implementation Approach

1.  **Refactor**: Ensure `Structure` can store calculation parameters (provenance).
2.  **QE Interface**: Implement `QECalculator`.
    -   Use `tprnfor=True` and `tstress=True`.
    -   Handle pseudopotential paths (environ variables or config).
3.  **Healing**: Implement `Healer` class.
    -   Wrap the `get_potential_energy()` call in a `try...except` block.
    -   On exception, call `healer.heal()`, update calculator, and retry.
4.  **Embedding**: Implement `embed_cluster` using `ase.build` tools.
5.  **Integration**: Update `Orchestrator` to call `Oracle.compute()`.

## 5. Test Strategy

### 5.1. Unit Tests
-   **Input Generation**: Verify `QECalculator` writes correct `pw.x` input files (check `control`, `system`, `electrons` sections).
-   **Healing Logic**: Mock an `Espresso` calculator that raises `EspressoError` once, then succeeds. Verify `Healer` changed the parameters.
-   **Embedding**: Verify `embed_cluster` creates a cell with correct dimensions and no atom overlap at boundaries.

### 5.2. Integration Tests (Mock QE)
-   **Mock Binary**: Create a dummy `pw.x` script that reads input and writes a valid XML/text output with random forces.
-   **Full Run**: Configure `Oracle` to use the mock binary. Run a batch of 10 structures. Verify they come back labeled.

### 5.3. Real DFT Test (Requires QE)
-   **Tiny System**: H2 molecule or single Si atom.
-   **Validation**: Run `Oracle.compute()` and check if energy is reasonable (e.g., cohesive energy of Si).
-   **Failure Test**: Intentionally set `mixing_beta=1.0` (unstable) for a hard system to trigger healing (if possible) or just verify error catching.
