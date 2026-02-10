# Cycle 03 User Acceptance Test (UAT)

## 1. Test Scenarios

### **UAT-03: Robust DFT Pipeline (Mocked/Small)**
*   **Goal**: Ensure that the DFT Oracle can execute calculations and recover from common convergence failures.
*   **Priority**: High
*   **Success Criteria**:
    *   The Oracle correctly creates input files (`pw.in`) with required flags (`tprnfor`, `tstress`).
    *   The Oracle correctly parses output files (`pw.out`) to extract Energy, Forces, and Stress.
    *   The Oracle attempts to retry (Self-Heal) at least once when a simulated SCF failure occurs.

## 2. Behavior Definitions (Gherkin)

### Scenario: Successful SCF Calculation
**GIVEN** a valid `config.yaml` with `oracle.type: qe` and a valid `command` (or mock binary)
**WHEN** the Oracle computes a simple structure (e.g., bulk Si)
**THEN** the returned structure should contain `energy` (float)
**AND** the returned structure should contain `forces` (Nx3 array)
**AND** the returned structure should contain `stress` (3x3 array or 6-element vector)

### Scenario: Self-Healing Mechanism
**GIVEN** a mock QE binary that fails on the first attempt (simulating non-convergence) but succeeds on the second
**WHEN** the Oracle computes a difficult structure (e.g., surface slab)
**THEN** the log should show "SCF Convergence Warning: Retrying with lower mixing_beta"
**AND** the final result should be successful (not an exception)
**AND** the second attempt's input file should have `mixing_beta < 0.7`
