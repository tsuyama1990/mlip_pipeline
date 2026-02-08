# Cycle 06: Validation & Orchestration UAT

## 1. Test Scenarios

### Scenario 06-01: Full System Integration
**Priority**: Critical
**Goal**: Verify the entire workflow.
**Description**:
1.  Run the CLI: `python -m mlip_autopipec run config.yaml`.
2.  Use Mock components for speed, but real logic for Orchestration.
3.  Check the `workdir` structure.
**Expected Outcome**:
-   `workdir/cycle_01` ... `cycle_N` directories created.
-   `report.html` summarizes the learning curve (RMSE vs Data Size).

### Scenario 06-02: Physical Validation (Phonons)
**Priority**: High
**Goal**: Verify potential stability.
**Description**:
1.  Train a potential on just equilibrium bulk.
2.  Validate it on a compressed (-10%) structure.
3.  The phonon dispersion might show imaginary modes (instability).
**Expected Outcome**:
-   The Validator reports `passed=False`.
-   The reason is "Imaginary Phonon Frequency at X point".

### Scenario 06-03: Elastic Constants Check
**Priority**: Medium
**Goal**: Verify mechanical properties.
**Description**:
1.  Compute elastic constants $C_{11}, C_{12}, C_{44}$ for cubic crystal.
2.  Compare with reference values (e.g., DFT or experiment).
**Expected Outcome**:
-   The calculated values are within 20% of the reference.
-   The Born stability criteria are satisfied.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Final Validation

  Scenario: Validate a stable potential
    Given a well-trained potential for Bulk Si
    When I request phonon validation
    Then no imaginary frequencies should be detected
    And the bulk modulus should be ~98 GPa

  Scenario: Reject an unstable potential
    Given a potential trained only on liquid data
    When I request phonon validation for the crystal
    Then imaginary frequencies should be detected
    And the potential should be marked as "Unstable"
```

## 3. Jupyter Notebook Validation (`tutorials/05_Full_Workflow.ipynb`)
-   **Run**: Use the Orchestrator API directly in the notebook to run 3 cycles.
-   **Visualize**:
    -   Plot Training Error vs Cycle.
    -   Plot Number of Data Points vs Cycle.
    -   Show the final validation report (HTML iframe).
