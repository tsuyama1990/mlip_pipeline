# Cycle 02: Structure Generator UAT

## 1. Test Scenarios

### Scenario 02-01: Bulk Generation
**Priority**: High
**Goal**: Verify basic structure generation.
**Description**:
Generate 50 bulk structures for a target system (e.g., Fe).
**Expected Outcome**:
-   The list contains valid `Structure` objects.
-   Most are perturbed (strained/rattled) from ideal positions.
-   The cell volumes vary within the specified range (e.g., +/- 10%).

### Scenario 02-02: Surface Generation
**Priority**: Medium
**Goal**: Verify capability to create surfaces for catalysis/adhesion.
**Description**:
Generate surfaces with indices (100), (110), (111).
**Expected Outcome**:
-   The z-axis length is significantly larger than x/y (vacuum padding).
-   The correct number of atoms for the specified supercell size.

### Scenario 02-03: Adaptive Policy (Mock Metrics)
**Priority**: Critical
**Goal**: Verify the generator reacts to system state.
**Description**:
1.  Feed a mock `training_metrics` where `force_rmse_surface` is high (e.g., 0.5 eV/A).
2.  Call `StructureGenerator.generate(n=100)`.
**Expected Outcome**:
-   The returned batch contains > 50% surface structures (based on tags/info).
-   Logs indicate "High surface error detected -> Boosting surface sampling".

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Structure Generation

  Scenario: Generate Bulk Strained Fe
    Given a configuration for element "Fe"
    When I request 10 structures from "BulkBuilder" with strain range 0.10
    Then all structures should have 1 or 2 atoms (primitive/conventional) scaled up to supercell
    And the volume of each structure should differ

  Scenario: Adaptive Policy triggers Surface Sampling
    Given the current model has high error on "surface" configurations
    When I request the next batch of 100 structures
    Then at least 60 structures should have the tag "surface"
```

## 3. Jupyter Notebook Validation (`tutorials/01_Structure_Gen.ipynb`)
-   **Visual Check**: Load the generated `extxyz` or `pckl` file. Use `ase.visualize.plot` to show a few examples.
-   **Distribution Check**: Plot a histogram of volumes/densities to confirm diversity.
