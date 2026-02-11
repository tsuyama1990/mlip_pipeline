# Cycle 08 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 8.1: Phonon Stability Check (Known Stable)
*   **Goal**: Verify the system correctly identifies a stable potential.
*   **Action**:
    1.  User updates `config.yaml` to include a `validator` section.
    2.  User uses a known stable potential (e.g., EMT for Cu).
    3.  User runs `pyacemaker validate`.
*   **Success Criteria**:
    *   The `validation_report.html` is generated.
    *   The report contains a Phonon Dispersion plot with no imaginary modes (all $\omega > -0.1$ THz).
    *   The report explicitly states "Status: STABLE".

### Scenario 8.2: Elastic Constants (Born Criteria)
*   **Goal**: Verify the system computes elastic constants and checks stability.
*   **Action**:
    1.  User runs validation on a cubic system (e.g., Al).
    2.  User inspects the report.
*   **Success Criteria**:
    *   The report lists $C_{11}, C_{12}, C_{44}$.
    *   The report shows "Passed Born Stability Check: Yes".
    *   The calculated Bulk Modulus matches the literature value within 10-20%.

### Scenario 8.3: Full Validation Report
*   **Goal**: Verify the generation of a comprehensive HTML report summarizing all cycles.
*   **Action**:
    1.  User runs the full Active Learning Loop (Cycles 1-7).
    2.  At the end of Cycle 7, the Validator is triggered.
    3.  User opens `validation_report.html` in a browser.
*   **Success Criteria**:
    *   The report contains:
        *   Training metrics (RMSE vs Epoch).
        *   Phonon Dispersion.
        *   EOS Curve (Energy vs Volume).
        *   Elastic Tensor table.
    *   The plots are interactive (if Plotly used) or high-quality PNGs.

## 2. Behavior Definitions (Gherkin Style)

### Feature: Phonon Validation
**Scenario**: Calculating phonon band structure
  **Given** a trained potential and a unit cell
  **When** the Validator runs the phonon calculation
  **Then** it should compute force constants
  **And** it should plot the dispersion relation along high-symmetry paths
  **And** it should flag any imaginary frequencies

### Feature: Elastic Validation
**Scenario**: Calculating elastic constants
  **Given** a cubic crystal structure
  **When** the Validator applies small strains
  **Then** it should compute the stress tensor
  **And** it should fit the stiffness matrix $C_{ij}$
  **And** it should check the Born stability criteria

### Feature: Reporting
**Scenario**: Generating final report
  **Given** a set of validation results and training logs
  **When** the ReportGenerator is invoked
  **Then** it should produce an HTML file
  **And** the file should aggregate all key metrics and plots
