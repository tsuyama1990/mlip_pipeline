# Cycle 03 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Basic DFT Calculation (Mocked)
**Priority**: High
**Goal**: Verify the Oracle interface works with ASE.
**Procedure**:
1.  Configure `oracle: type: mock` (or `qe` with a dummy executable script).
2.  Provide a generated structure from Cycle 02.
3.  Run the Oracle step.
**Expected Result**:
*   The system produces a labeled dataset (e.g., `labeled_data.pckl`).
*   The structure in the dataset has `info['energy']` and `arrays['forces']`.

### Scenario 2: Error Recovery
**Priority**: Medium
**Goal**: Verify self-correction logic.
**Procedure**:
1.  Use a mock script for `pw.x` that fails on the first run (simulating SCF error) and passes on the second run (when `mixing_beta` is changed).
2.  Run the Oracle.
**Expected Result**:
*   Log shows: "SCF convergence failed. Reducing mixing beta to 0.3".
*   Final result is successful.

### Scenario 3: Periodic Embedding
**Priority**: Medium
**Goal**: Verify non-periodic inputs are handled.
**Procedure**:
1.  Manually create a structure with `pbc=False` (simulating a cluster cut from MD).
2.  Pass it to the Oracle.
**Expected Result**:
*   The system does not crash.
*   The input file generated for DFT has a defined cell (lattice parameters) and valid atomic positions.

## 2. Behavior Definitions

```gherkin
Feature: DFT Calculation

  Scenario: Computing energy for a valid structure
    GIVEN a structure "Si_bulk"
    AND a configured QE Oracle
    WHEN "compute" is called
    THEN the structure should be updated with Energy and Forces
    AND the output file should be saved in "active_learning/iter_XXX/dft_calc"

  Scenario: Recovering from SCF failure
    GIVEN a difficult structure
    WHEN the DFT calculation fails with "convergence not achieved"
    THEN the Oracle should modify the input parameters (smearing/mixing)
    AND resubmit the job
    AND return the result from the successful run
```
