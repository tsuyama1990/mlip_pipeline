# Cycle 03: User Acceptance Test (UAT) Plan

## 1. Test Scenarios

### Scenario 3.1: Basic DFT Calculation (Mock)
**Priority**: Critical
**Description**: Verify that the Oracle interface works correctly with the Mock implementation. This ensures that the system can proceed without an actual DFT code installed.

**Jupyter Notebook**: `tutorials/02_oracle_test.ipynb`
1.  Initialize `MockOracle` via `ComponentFactory`.
2.  Create a list of 5 random structures (using `StructureGenerator` from Cycle 02).
3.  Call `oracle.compute(structures)`.
4.  Assert that the returned list has 5 `CalculationResult` objects.
5.  Verify that `energy` is a float and `forces` is a numpy array of shape (N_atoms, 3).
6.  Verify that `converged` is True for all.

### Scenario 3.2: Quantum Espresso Input Generation
**Priority**: High
**Description**: Verify that the system generates correct input files for Quantum Espresso, specifically handling k-point spacing.

**Jupyter Notebook**: `tutorials/02_oracle_test.ipynb`
1.  Initialize `QEOracle` with `kspacing=0.04`.
2.  Create a silicon primitive cell (2 atoms).
3.  Call `oracle._prepare_input(structure)` (or inspect generated file).
4.  Assert that the generated k-point grid is dense (e.g., 6x6x6 or more).
5.  Create a silicon supercell (54 atoms).
6.  Call `oracle._prepare_input(structure)`.
7.  Assert that the generated k-point grid is sparse (e.g., 2x2x2).
8.  This confirms that k-spacing logic adapts to cell size.

### Scenario 3.3: Periodic Embedding
**Priority**: High
**Description**: Verify that large structures are correctly cut into smaller, periodic clusters for efficient DFT.

**Jupyter Notebook**: `tutorials/02_oracle_test.ipynb`
1.  Create a large 5x5x5 supercell of Aluminum (500 atoms).
2.  Introduce a vacancy at the center atom.
3.  Call `embedding.embed(structure, center_index=250, radius=6.0)`.
4.  Assert that the returned structure has significantly fewer atoms (e.g., ~50-100).
5.  Assert that the structure is periodic.
6.  Visualize the result to confirm the vacancy is at the center.

### Scenario 3.4: Error Handling & Self-Correction
**Priority**: Medium
**Description**: Verify that the system attempts to recover from SCF convergence failures.

**Jupyter Notebook**: `tutorials/02_oracle_test.ipynb`
1.  (Requires mocking `ase.calculators.espresso.Espresso.get_potential_energy` to raise `QEError`).
2.  Mock the calculator to fail on the first attempt but succeed on the second.
3.  Call `oracle.compute([structure])`.
4.  Assert that the system logs a warning ("SCF failed, retrying with mixing_beta=0.3").
5.  Assert that the final result is `converged=True`.

## 2. Behavior Definitions

### Mock Oracle Behavior
**GIVEN** a list of structures
**WHEN** `MockOracle.compute` is called
**THEN** it should return energies calculated via a simple Lennard-Jones potential
**AND** forces should match the analytical gradient of that potential.

### QE Input Generation
**GIVEN** a metallic system (Fe)
**AND** `smearing` is not explicitly set in config
**WHEN** `QEOracle` prepares the input
**THEN** it should automatically set `occupations='smearing'`, `smearing='mv'`, and `degauss=0.01` (or reasonable defaults).

### Periodic Embedding Logic
**GIVEN** an atom at the edge of the simulation box (near 0,0,0)
**WHEN** `embedding.embed` is called
**THEN** it should correctly wrap around the periodic boundaries to include neighbors from the other side of the box.
