# Cycle 02 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 02-01: Bulk Structure Generation
- **Priority**: High
- **Description**: Generate simple bulk supercells with thermal noise.
- **Steps**:
  1. Create a python script importing `StructureBuilder`.
  2. Configure it for a $2 \times 2 \times 2$ Al supercell with `rattle=0.1`.
  3. Generate 5 structures.
- **Expected Result**: 5 structures are returned. Visualizing them shows slight disorder.

### Scenario 02-02: Defect Introduction
- **Priority**: Medium
- **Description**: Create a vacancy in a supercell.
- **Steps**:
  1. Generate a perfect supercell.
  2. Apply `create_vacancy`.
- **Expected Result**: Atom count decreases by 1.

### Scenario 02-03: Reproducibility
- **Priority**: Critical
- **Description**: Ensure identical seeds produce identical structures.
- **Steps**:
  1. Run Builder with `seed=42`. Save positions.
  2. Run Builder again with `seed=42`.
- **Expected Result**: Positions match exactly.

## 2. Behavior Definitions

```gherkin
Feature: Structure Generation

  Scenario: Generate Rattled Structures
    GIVEN a base structure of Aluminum
    AND a generator config with "rattle_stdev: 0.1"
    WHEN the generator produces 5 samples
    THEN the result should be a list of 5 Atom objects
    AND the positions should vary between samples
```
