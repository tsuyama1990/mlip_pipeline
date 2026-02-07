# Cycle 02 User Acceptance Testing (UAT)

## 1. Test Scenarios

### SCENARIO 01: Random Structure Generation
**Priority**: High
**Goal**: Verify that the Structure Generator can create physically valid random configurations.

**Steps**:
1.  Create a configuration file:
    ```yaml
    generator:
      type: random
      count: 10
      composition:
        Fe: 0.5
        Pt: 0.5
      num_atoms: 8
      min_distance: 2.0
      cell_size: 5.0
    ```
2.  Run CLI command: `mlip-pipeline generate --config config.yaml --output generated.jsonl`.
3.  Check output file `generated.jsonl`.
4.  Verify (using a small script or ASE) that no pair distance is < 2.0 Å.

### SCENARIO 02: Dataset Handling
**Priority**: Medium
**Goal**: Verify that large datasets can be written and read without memory issues.

**Steps**:
1.  Run a script to generate 1000 dummy structures and append them to `dataset.jsonl`.
2.  Read the file sequentially.
3.  Verify that all 1000 structures are recovered with correct atomic numbers and positions.

## 2. Behavior Definitions

### Feature: Structure Validation
**Scenario**: Generating Random FePt Alloy
  **Given** a request for 10 random FePt structures with minimum distance 2.0 Å
  **When** the generator runs
  **Then** it should produce 10 structures
  **And** each structure should contain 50% Fe and 50% Pt
  **And** no two atoms in any structure should be closer than 2.0 Å (considering Periodic Boundary Conditions)
