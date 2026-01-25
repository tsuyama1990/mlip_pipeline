# Cycle 02 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: DFT Input Generation Verification
*   **ID**: UAT-02-01
*   **Priority**: High
*   **Description**: Verify that the system generates valid Quantum Espresso input files for various crystal structures.
*   **Success Criteria**:
    *   Input file contains all mandatory flags (`calculation='scf'`, `tprnfor=.true.`).
    *   Pseudopotentials are correctly assigned from the configuration.
    *   K-point grid is reasonable (e.g., $4 \times 4 \times 4$ for a typical solid).

### Scenario 2: Self-Healing Mechanism
*   **ID**: UAT-02-02
*   **Priority**: Critical
*   **Description**: Simulate a convergence failure and observe the system's attempt to recover.
*   **Success Criteria**:
    *   System logs a warning "DFT convergence failed, retrying with strategy...".
    *   The second attempt uses a smaller `mixing_beta`.
    *   The final result is reported as success (assuming the mock allows it).

### Scenario 3: Periodic Embedding correctness
*   **ID**: UAT-02-03
*   **Priority**: High
*   **Description**: Verify that the embedding logic correctly cuts out a local environment.
*   **Success Criteria**:
    *   Given a 1000-atom supercell with one perturbed atom, the embedded cell should contain approx 50-100 atoms (depending on cutoff).
    *   The distance between the perturbed atom and its neighbors in the new cell must match the original.

## 2. Behavior Definitions

```gherkin
Feature: Oracle (DFT Automation)

  As a system architect
  I want the DFT engine to be self-correcting
  So that the automated loop doesn't stop due to minor convergence issues

  Scenario: Recover from Convergence Failure
    GIVEN a structure that is hard to converge
    AND a QERunner configured with self-healing
    WHEN I request a single point calculation
    AND the first attempt returns "convergence not achieved"
    AND the second attempt returns "JOB DONE"
    THEN the calculation should be marked as successful
    AND the log should show "Retry 1/3: Reduced mixing_beta"

  Scenario: Periodic Embedding
    GIVEN a large supercell with a central defect
    WHEN I apply periodic embedding with R_cut=5.0
    THEN the resulting structure should have fewer atoms than the original
    AND the resulting cell should be large enough to avoid self-interaction of the defect
```
