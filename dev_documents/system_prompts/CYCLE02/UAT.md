# User Acceptance Test (UAT): Cycle 02

## 1. Test Scenarios

### Scenario 02-01: The Oracle Speaks (Priority: High)
**Objective**: Verify that the system can perform a successful DFT calculation and return labelled data.

**Description**:
The user provides a simple atomic structure (e.g., a bulk Silicon crystal) and requests its energy and forces. The system should generate the input file for Quantum Espresso, execute it (or mock it), and parse the results back into a format the system understands.

**User Journey**:
1.  The user prepares a `config.yaml` specifying the path to `pw.x` and pseudopotentials.
2.  The user runs a script (or the main system) that triggers the Oracle on a single structure.
3.  The system logs "Submitting job to pw.x...".
4.  The system logs "Calculation converged in 12 steps."
5.  The system outputs the Energy (-156.2 eV) and Forces (near zero for equilibrium).
6.  The user verifies the values are physically reasonable (if running Real Mode).

**Success Criteria**:
*   The system correctly finds the `pw.x` executable.
*   The input file `pw.in` is generated with the correct atomic positions.
*   The output object contains `energy`, `forces`, and `stress`.

### Scenario 02-02: The Self-Healing Sage (Priority: High)
**Objective**: Verify the "Self-Healing" capability when a calculation fails.

**Description**:
In this scenario, we force a convergence failure. In "Mock Mode", this is done by instructing the mock runner to throw an error on the first attempt. In "Real Mode", one might use a pathological system (e.g., magnetic iron with bad initial guess). We want to see the system detect the error and retry automatically.

**User Journey**:
1.  The system starts a calculation.
2.  The log shows "Error: convergence not achieved."
3.  The log shows "WARN: Attempt 1 failed. Retrying with reduced mixing_beta = 0.3..."
4.  The system restarts the calculation.
5.  The calculation succeeds on the second try.
6.  The user sees a successful result, oblivious to the fact that it almost failed.

**Success Criteria**:
*   The logs clearly show the retry attempt and the changed parameter.
*   The final result is returned successfully, not an exception.

## 2. Behavior Definitions (Gherkin)

### Feature: DFT Calculation

```gherkin
Feature: Quantum Espresso Interface

  Scenario: Successful Single Point Calculation
    GIVEN a valid atomic structure "Si_bulk"
    AND the DFT configuration points to valid pseudopotentials
    WHEN the Oracle computes properties for "Si_bulk"
    THEN an input file "pw.in" should be generated
    AND the k-points should be set based on "kspacing"
    AND the system should return an Atoms object with attached Calculator
    AND the returned object should have "energy", "forces", and "stress"

  Scenario: Recover from SCF Convergence Failure
    GIVEN an atomic structure that is difficult to converge
    WHEN the Oracle computes properties
    AND the first execution of "pw.x" fails with "convergence not achieved"
    THEN the system should NOT raise an exception immediately
    AND the system should generate a new input file
    AND the new input file should have "mixing_beta" lower than the default
    AND the system should retry the execution
    AND if the second execution succeeds, the result should be returned
```
