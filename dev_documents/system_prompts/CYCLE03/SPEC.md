# Cycle 03 Specification: Oracle & DFT Interface

## 1. Summary
This cycle is dedicated to building the "Oracle" component, which provides the ground-truth data (energy, forces, stresses) essential for training accurate MLIPs. We will implement `DFTManager`, a robust interface to Quantum Espresso (QE) via the Atomic Simulation Environment (ASE). Key features include "Self-Healing" capabilities to automatically recover from common SCF convergence failures and "Periodic Embedding" to efficiently calculate forces on local clusters extracted from large MD snapshots.

## 2. System Architecture

The following file structure will be created/modified. Files in **bold** are the primary deliverables for this cycle.

```ascii
src/
└── mlip_autopipec/
    ├── implementations/
    │   └── **oracle/**
    │       ├── **__init__.py**
    │       ├── **dft_manager.py**      # Main DFT Interface
    │       └── **embedding.py**        # Periodic Embedding Logic
    └── utils/
        └── **mock_pw.py**              # Mock Binary for Integration Tests
```

## 3. Design Architecture

### 3.1. DFTManager
The `DFTManager` class implements the `BaseOracle` interface. It is configured via `OracleConfig`.
-   **ASE Integration**: It uses `ase.calculators.espresso.Espresso` to handle input generation and output parsing.
-   **Self-Healing**: It wraps the calculation execution in a retry loop. If a `CalculatoryError` occurs (e.g., "convergence not achieved"), it adjusts parameters (e.g., reducing `mixing_beta`, increasing `smearing_width`) and retries up to `max_retries` times.

### 3.2. Periodic Embedding
The `PeriodicEmbedding` utility class (or function) handles the extraction of local environments.
-   **Concept**: When an MD simulation detects a high-uncertainty atom, we cannot afford to recalculate the entire MD box (thousands of atoms). Instead, we carve out a cluster (radius $R_{cut} + R_{buffer}$) and place it in a smaller, isolated periodic box.
-   **Implementation**: This logic ensures that the forces on the central atoms are physically meaningful (bulk-like) despite the artificial boundary, by leveraging the short-ranged nature of the MLIP.

## 4. Implementation Approach

### Step 1: Periodic Embedding
Implement `embedding.py`.
-   Input: Large `Structure`, list of `centre_atom_indices`, `cutoff`, `buffer`.
-   Output: List of small `Structure` objects (clusters in boxes).
-   Algorithm: Use ASE's neighbour list to find atoms within radius. Construct a new cell that fits the cluster with vacuum padding (or periodic repetition if applicable).

### Step 2: DFTManager Implementation
Implement `DFTManager` in `dft_manager.py`.
-   Initialise `Espresso` calculator with user-provided pseudopotentials and K-points.
-   Implement the `compute(structures)` method to iterate over structures and run calculations.
-   Add the retry logic: `try... except... adjust parameters... retry`.

### Step 3: Mock Binary (Test Helper)
Create `mock_pw.py`. This script mimics the behaviour of `pw.x` (Quantum Espresso binary).
-   It reads the input file (stdin or file).
-   It writes a valid QE output to stdout, containing dummy energy and forces.
-   This allows us to test the *entire* `DFTManager` pipeline (input generation -> execution -> parsing) in CI without installing QE.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Embedding Logic**: Verify that extracted clusters contain the correct number of atoms and that PBCs are handled correctly.
-   **Input Generation**: Verify that `DFTManager` generates valid PWscf input files (check for `tprnfor=.true.`, `tstress=.true.`).

### 5.2. Integration Testing (using Mock Binary)
-   **Full Cycle**: Point `DFTManager` to use `python mock_pw.py` as the command.
-   Run `compute(structure)`.
-   Assert that the returned structure has `energy`, `forces`, and `stress` attached.
-   **Self-Healing**: Modify `mock_pw.py` to fail on the first attempt and succeed on the second. Verify that `DFTManager` retries and eventually succeeds.
