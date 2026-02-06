# Cycle 08 Specification: Scale-up & Production (EON & Validation)

## 1. Summary
This final cycle completes the system by adding capabilities for long-timescale simulations and rigorous physical validation. We integrate **EON** (for Adaptive Kinetic Monte Carlo) to study diffusion and rare events that are inaccessible to standard MD. We also implement **Periodic Embedding**, a sophisticated technique to cut small, DFT-computable clusters from large MD simulations while preserving valid boundary conditions. Finally, we add advanced validation metrics (Phonon Dispersion and Elastic Constants) to ensure the potentials are truly production-ready.

## 2. System Architecture

### 2.1. File Structure

```
mlip-pipeline/
├── src/
│   └── mlip_autopipec/
│       ├── infrastructure/
│       │   ├── **eon/**
│       │   │   ├── **__init__.py**
│       │   │   └── **adapter.py**      # [NEW] EonExplorer implementation
│       │   └── **phonopy/**
│       │       └── **adapter.py**      # [NEW] Phonon validation
│       └── utils/
│           └── **embedding.py**        # [NEW] Periodic Embedding logic
└── tests/
    └── unit/
        └── **test_embedding.py**       # [NEW] Geometric tests
```

## 3. Design Architecture

### 3.1. `EonExplorer` Class
Implements `BaseExplorer` but focuses on *Rare Event Search*.
*   **Wrapper**: Wraps the `eonclient` binary.
*   **Task**: Run aKMC (saddle point searches) starting from a given structure.
*   **Output**: A list of saddle points or new minima found during the search.
*   **Halt Logic**: Similar to LAMMPS, EON can trigger a halt if the potential uncertainty is high during the NEB/Dimer search.

### 3.2. Periodic Embedding (`utils/embedding.py`)
This is a critical algorithm for scaling Active Learning.
*   **Problem**: We run MD on 10,000 atoms. We find 1 "confusing" atom. We cannot run DFT on 10,000 atoms.
*   **Solution**:
    1.  Select the central atom $i$.
    2.  Select neighbors within $R_{cut} + R_{buffer}$.
    3.  Construct a **minimal orthorhombic box** that encloses these neighbors.
    4.  Apply periodic boundary conditions to this small box.
    5.  Pass this "embedded cluster" (approx 50-100 atoms) to the Oracle.
    6.  **Force Masking**: Only train on the forces of the central atom $i$, ignoring the boundary atoms (which see a fake periodic image).

### 3.3. Advanced Validation
*   **Phonons**: Use `phonopy` to calculate band structures. Check for imaginary frequencies ($\omega^2 < 0$) which indicate dynamic instability.
*   **Elasticity**: Calculate $C_{11}, C_{12}, C_{44}$. Check Born stability criteria.

## 4. Implementation Approach

1.  **Periodic Embedding**: Implement `utils/embedding.py`.
    *   Use `ase.neighborlist` to find neighbors.
    *   Logic to create a padded box.
    *   Logic to wrap positions correctly.
2.  **EON Adapter**: Create `infrastructure/eon/adapter.py`.
    *   Generate `config.ini` for EON.
    *   Provide a Python driver (`pace_driver.py`) that EON calls to get Energy/Forces from our `.yace` potential.
3.  **Phonopy Adapter**: Create `infrastructure/phonopy/adapter.py`.
    *   Generate displacements.
    *   Calculate forces (using the MLIP).
    *   Compute force constants and band structure.
4.  **Final Polish**: Ensure CLI supports all these new features.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Embedding Geometry**:
    *   Create a known structure (e.g., FCC crystal).
    *   Pick an atom.
    *   Run `embed_cluster()`.
    *   **Assert**: The resulting `Atoms` object is small (< 100 atoms).
    *   **Assert**: The central atom has the same local environment (neighbors) as in the bulk.

### 5.2. Integration Testing
*   **EON Mock**: Mock the `eonclient` execution. Verify it returns new structures.
*   **Phonopy Mock**: Verify the data flow between `Validator` and `Phonopy`.

### 5.3. Final UAT (The Tutorial)
*   Run the "Fe/Pt on MgO" tutorial notebook in "CI Mode".
*   This tests the entire stack: Config -> Orchestrator -> Hybrid Potential -> LAMMPS -> EON -> Validation.
