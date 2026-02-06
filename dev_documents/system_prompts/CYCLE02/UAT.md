# Cycle 02 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 2.1: DFT Input Generation
**Priority**: Critical
**Description**: Verify that the system generates correct Quantum Espresso input files for various crystal structures.
**Steps**:
1.  Define a `OracleConfig` with specific pseudopotentials and kspacing.
2.  Pass a Bulk Silicon structure (2 atoms) and a large Supercell (54 atoms) to the Oracle.
3.  Inspect the generated input files (mocked execution).
4.  **Check**: Does the k-point grid decrease as the cell size increases (preserving constant density)?
5.  **Check**: Are `tprnfor` and `tstress` enabled?

### Scenario 2.2: Self-Healing Mechanism
**Priority**: High
**Description**: Verify that the Oracle attempts to fix convergence errors automatically.
**Steps**:
1.  Configure the Mock Oracle to fail the first 2 attempts with "Convergence NOT achieved".
2.  Run `oracle.label()`.
3.  Check logs.
4.  **Expectation**:
    *   Log: "SCF failed. Retrying with mixing_beta=0.3..."
    *   Log: "SCF failed. Retrying with smearing='methfessel-paxton'..."
    *   Final Result: Success (simulated).

### Scenario 2.3: Handling Invalid Structures
**Priority**: Medium
**Description**: Ensure the system handles catastrophic failures gracefully (e.g., atoms too close).
**Steps**:
1.  Create a structure with two atoms at 0.1 Å distance (nuclear fusion).
2.  Run `oracle.label()`.
3.  Expect the DFT code (or Mock) to crash or return error.
4.  **Expectation**: The Oracle should catch the exception, log a warning "Structure failed validation," and return an empty result or marked failure, **without crashing the main pipeline**.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: DFT Labeling

  Scenario: Correct K-Point Scaling
    Given a target k-spacing of 0.04 1/A
    When I submit a small unit cell (5A)
    Then the generated K-points should be roughly 5x5x5
    When I submit a large supercell (20A)
    Then the generated K-points should be roughly 1x1x1

  Scenario: Automatic Error Recovery
    Given a structure that is hard to converge
    When the standard DFT calculation fails
    Then the Oracle should automatically retry with reduced mixing beta
    And if successful, return the energy and forces
```
