# Cycle 06 User Acceptance Testing (UAT)

## 1. Test Scenarios

### SCENARIO 01: Validation Report Generation
**Priority**: High
**Goal**: Verify the system generates a comprehensive validation report after training.

**Steps**:
1.  Use a pre-trained potential (or LJ as proxy).
2.  Configure `ValidatorConfig`:
    ```yaml
    validator:
      type: all
      phonon_supercell: [2, 2, 2]
    ```
3.  Run CLI: `mlip-pipeline validate --config config.yaml --potential potential.yace`.
4.  Open `validation_report.html` in a browser.
5.  Check for Parity Plots, Phonon Dispersion Curves, and Elastic Constants table.

### SCENARIO 02: EON (aKMC) Integration
**Priority**: Medium
**Goal**: Verify that EON can run using the trained potential.

**Steps**:
1.  Configure `DynamicsConfig` for EON:
    ```yaml
    dynamics:
      type: eon
      temperature: 300
    ```
2.  Run CLI: `mlip-pipeline kcmc --config config.yaml`.
3.  Check logs for EON execution ("Saddle point search started...").
4.  Verify that `processed_events.dat` or similar output is generated.

## 2. Behavior Definitions

### Feature: Potential Validation
**Scenario**: Validating a Stable Crystal
  **Given** a potential trained on a stable FCC structure
  **When** the Validator runs the phonon stability test
  **Then** it should calculate the phonon band structure
  **And** it should find no significant imaginary frequencies (indicating instability)
  **And** it should mark the validation result as "PASS"
  **And** it should generate an HTML report with the band structure plot
