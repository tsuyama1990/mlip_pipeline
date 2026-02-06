# Cycle 03 UAT: The Explorer

## 1. Test Scenarios

### Scenario 3: Initial Structure Generation
**Priority**: High
**Objective**: Verify that the system can bootstrap itself from zero data by generating valid starting structures.

**Steps**:
1.  **Preparation**:
    *   Config: `explorer.initial_exploration_type = "random"`.
    *   Input: Composition "MgO".
2.  **Execution**:
    *   Run the pipeline.
3.  **Verification**:
    *   Check output: The system should generate a set of MgO structures (Rock salt prototype, potentially distorted).
    *   **Physics Check**: Load the generated `Atoms` in a notebook. Check nearest neighbor distance. It should not be < 1.0 Angstrom (no fusion).

## 2. Behavior Definitions

```gherkin
Feature: Adaptive Exploration

  Scenario: Generating Equation of State samples
    GIVEN a perfect MgO crystal structure
    AND a "StrainPolicy" configured with range +/- 10%
    WHEN the generator is invoked
    THEN it should produce a list of structures with varying volumes
    AND the atomic fractional coordinates should be preserved (affine transformation)
```
