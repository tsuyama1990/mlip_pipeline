# Cycle 03 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 03-01: Generate Valid QE Input
- **Priority**: High
- **Description**: Verify that the system generates a correct `input.pwi` file.
- **Steps**:
  1. Initialize `QERunner` with a config.
  2. Pass an Al FCC structure.
  3. Inspect the generated file.
- **Expected Result**: Contains `ATOMIC_POSITIONS`, `CELL_PARAMETERS`, and correct k-points (e.g., not 1x1x1 for a primitive cell).

### Scenario 03-02: Parse Successful Output
- **Priority**: High
- **Description**: Verify parsing of energy and forces.
- **Steps**:
  1. Provide a sample `pw.out` file.
  2. Run `QEOutputParser`.
- **Expected Result**: Returns a dict with `energy` (float) and `forces` (Nx3 array).

### Scenario 03-03: Auto-Recovery from Divergence
- **Priority**: Medium
- **Description**: System should retry with different params if SCF fails.
- **Steps**:
  1. Use a Mock Runner that fails the first 2 times (simulating convergence error) and succeeds the 3rd time.
  2. Call `runner.calculate()`.
- **Expected Result**: The call returns successfully. Logs show "Retrying with mixing_beta=0.3" etc.

## 2. Behavior Definitions

```gherkin
Feature: DFT Oracle

  Scenario: Run Static Calculation
    GIVEN a structure of 4 Silicon atoms
    WHEN the Oracle executes a calculation
    THEN it should produce Energy and Forces
    AND the status should be "Completed"

  Scenario: Handle Convergence Failure
    GIVEN a hard-to-converge structure
    WHEN the Oracle encounters "convergence NOT achieved"
    THEN it should automatically retry with reduced mixing beta
    AND eventually succeed or report "Max Retries Exceeded"
```
