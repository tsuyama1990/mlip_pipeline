# User Acceptance Testing (UAT): Cycle 03

## 1. Test Scenarios

Cycle 03 is about ensuring the system can reliably generate ground-truth data.

### Scenario 3.1: Standard DFT Calculation
-   **ID**: UAT-C03-01
-   **Priority**: Critical
-   **Description**: Run a standard static calculation (SCF) on a simple structure (e.g., Silicon unit cell).
-   **Success Criteria**:
    -   `pw.x` executes without error.
    -   The system parses the output and returns a `DFTResult` object.
    -   The `DFTResult` contains energy, forces (N x 3 array), and stress.
    -   Forces on a perfect crystal should be near zero (< 0.01 eV/A).

### Scenario 3.2: Self-Healing (The "Zombie" Calculation)
-   **ID**: UAT-C03-02
-   **Priority**: High
-   **Description**: We intentionally induce a convergence failure (e.g., by setting a ridiculously high mixing beta or low electron temperature for a metal). The system should automatically fix it.
-   **Success Criteria**:
    -   The logs show "SCF Failure detected. Applying fix: Reduce Mixing Beta".
    -   The system retries the calculation.
    -   The final result is reported as "Converged".

### Scenario 3.3: Automatic K-Point Grid
-   **ID**: UAT-C03-03
-   **Priority**: Medium
-   **Description**: The user defines `kspacing` instead of manual K-points.
-   **Success Criteria**:
    -   Run on a small unit cell (e.g., 5 Angstrom) -> High K-points (e.g., 6x6x6).
    -   Run on a large supercell (e.g., 20 Angstrom) -> Low K-points (e.g., 2x2x2 or 1x1x1).
    -   Verify the input file `pw.in` reflects this dynamic sizing.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: DFT Oracle and Self-Healing

  Background:
    Given the config has a valid path to "pw.x"
    And valid pseudopotentials are provided

  Scenario: Run a successful static calculation
    Given a Silicon primitive cell structure
    When I request a DFT calculation
    Then the process should finish successfully
    And the result should contain Energy
    And the result should contain Forces with shape (2, 3)

  Scenario: Recover from SCF convergence failure
    Given a difficult metallic structure
    And I force the initial parameters to be unstable (mixing_beta=0.9)
    When I request a DFT calculation
    Then the system should detect "convergence not achieved"
    And the system should retry with "mixing_beta=0.7"
    And the final status should be "Converged"

  Scenario: K-space density consistency
    Given a target k-spacing of 0.2 inverse Angstroms
    When I generate inputs for a 10.0 Angstrom cubic cell
    Then the K-points should be "4 4 4" (approx 2 * pi / (10 * 0.2) = 3.14 -> 4)
```
