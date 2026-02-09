# Cycle 02: User Acceptance Test (UAT) Plan

## 1. Test Scenarios

### Scenario 2.1: Initial Structure Generation
**Priority**: Critical
**Description**: Verify that the system can generate a diverse initial dataset for training. This is crucial for the "Cold Start" of the Active Learning loop.

**Jupyter Notebook**: `tutorials/01_structure_generation.ipynb`
1.  Initialize `StructureGenerator` with `config.yaml` specifying `composition: "MgO"`.
2.  Call `generator.generate_initial_structures(n=50)`.
3.  Visualize the output using `ase.visualize.plot`.
4.  Assert that the list contains:
    *   Bulk structures (various lattice constants).
    *   Surface slabs (e.g., (100), (110)).
    *   Clusters (optional, if configured).
5.  Check for duplicate structures (using `StructureMatcher` or simple fingerprinting).

### Scenario 2.2: Adaptive Exploration Policy (Mock)
**Priority**: Medium
**Description**: Verify that the exploration policy adapts sampling parameters based on material features.

**Jupyter Notebook**: `tutorials/01_structure_generation.ipynb`
1.  Create `MaterialFeatures` for a high-melting-point material ($T_m=3000K$).
2.  Call `policy.decide_strategy(features)`.
3.  Assert that `strategy.temperature_max` is high (e.g., > 2500K).
4.  Create features for a low-melting-point material ($T_m=300K$).
5.  Assert that `strategy.temperature_max` is low (e.g., < 400K).

### Scenario 2.3: Defect Generation
**Priority**: Medium
**Description**: Verify that defects are correctly introduced without breaking periodicity or creating unphysical overlaps.

**Jupyter Notebook**: `tutorials/01_structure_generation.ipynb`
1.  Create a perfect MgO crystal.
2.  Call `generator.create_vacancy(structure, concentration=0.01)`.
3.  Assert that 1 atom is removed (for a 100-atom cell).
4.  Call `generator.apply_strain(structure, strain=[0.05, 0, 0, 0, 0, 0])`.
5.  Assert that the cell parameter $a$ has increased by 5%.

## 2. Behavior Definitions

### Initial Generation
**GIVEN** a configuration for "FePt"
**WHEN** `generate_initial_structures` is called
**THEN** the output should contain Fe-rich, Pt-rich, and stoichiometric FePt structures
**AND** the `provenance` field should indicate "random_bulk", "random_surface", etc.

### Policy Adaptation
**GIVEN** a material identified as "Metal" (zero band gap)
**WHEN** the policy is evaluated
**THEN** the `mc_ratio` (Monte Carlo swap ratio) should be non-zero (to explore alloy configurations)
**AND** `strain_range` should be moderate.

### Defect Safety
**GIVEN** a structure with 2 atoms
**WHEN** a vacancy is requested
**THEN** the generator should raise a warning or error (cannot remove 50% of atoms safely in small cell)
**OR** should return the original structure if impossible.
