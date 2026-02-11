# Cycle 02 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 2.1: Cold Start Generation
*   **Priority**: High
*   **Goal**: Verify system can start from zero data.
*   **Action**:
    1.  Configure `generator.strategy: ADAPTIVE`.
    2.  Run `Orchestrator` cycle 1.
*   **Expectation**:
    *   `work_dir/iter_001/candidates/` contains multiple `structure_*.xyz` files.
    *   Files contain valid atomic positions (checked with `ase.io.read`).

### Scenario 2.2: Distortion Magnitude
*   **Priority**: Medium
*   **Goal**: Verify `distortion_magnitude` config affects output.
*   **Action**:
    1.  Run with `distortion_magnitude: 0.0`.
    2.  Run with `distortion_magnitude: 0.5`.
*   **Expectation**:
    *   Case 1: Output structures are identical to input (perfect lattice).
    *   Case 2: Output structures are heavily distorted (large displacement).

### Scenario 2.3: Supercell Expansion
*   **Priority**: Medium
*   **Goal**: Verify supercell handling.
*   **Action**:
    1.  Configure `supercell_matrix: [2, 2, 2]`.
    2.  Run generator on a 2-atom unit cell.
*   **Expectation**:
    *   Resulting structures have 16 atoms (2 * 2^3).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Structure Generation

  Scenario: Adaptive Policy selects strategy
    Given the current cycle is 0
    When the generator is invoked
    Then the policy should select "COLD_START"
    And the generator should produce "10" candidate structures

  Scenario: Generator applies physical constraints
    Given a bulk structure "MgO"
    When the generator applies a strain of "0.1"
    Then the cell volume should change by at most "15%"
    And the atom count should remain constant
```
