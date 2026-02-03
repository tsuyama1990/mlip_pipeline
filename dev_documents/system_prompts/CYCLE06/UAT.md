# Cycle 06 UAT: Advanced Physics & Deployment

## 1. Test Scenarios

### Scenario 01: The kMC Bridge
**Priority**: Medium
**Description**: Run a saddle point search using EON + Pacemaker.
**Objective**: Verify the `pace_driver.py` works.

**Steps**:
1.  Setup an EON directory with `config.ini` pointing to `pace_driver.py`.
2.  Provide a reactant structure.
3.  Run `eonclient`.
4.  **Expected Result**:
    -   EON runs multiple steps.
    -   It finds a saddle point (Process found).
    -   It does not crash with "Communication Error".

### Scenario 02: Phonon Stability
**Priority**: High
**Description**: Validate a stable crystal.
**Objective**: Verify Phonopy integration.

**Steps**:
1.  Train a potential for Bulk Silicon.
2.  Run the Validation suite.
3.  **Expected Result**:
    -   The generated phonon band structure shows no imaginary modes (no curves going below zero frequency).
    -   The report is generated as `validation_report.html`.

### Scenario 03: The Release
**Priority**: Low
**Description**: Package the potential.
**Objective**: Verify artifact creation.

**Steps**:
1.  Run `pyacemaker package --potential potentials/final.yace --tag v1.0`.
2.  **Expected Result**:
    -   A file `release_v1.0.zip` is created.
    -   Unzipping it reveals the potential and a `manifest.json` with today's date.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Advanced Validation & Deployment

  Scenario: Detecting Instability
    Given a potential that overfits
    When Phonon validation is run
    Then imaginary frequencies should be detected
    And the Validation Report should be marked "FAIL"

  Scenario: Production Packaging
    Given a validated potential
    When the user requests a release package
    Then a ZIP file should be created
    And it must contain the potential, the manifest, and the license
```
