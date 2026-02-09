# Cycle 08: User Acceptance Test (UAT) Plan

## 1. Test Scenarios

### Scenario 8.1: Validation of a Stable Potential (Mock)
**Priority**: Critical
**Description**: Verify that a physically sound potential passes all validation checks.

**Jupyter Notebook**: `tutorials/07_validation_test.ipynb`
1.  Initialize `StandardValidator` with default thresholds.
2.  Mock `PhononCalc` to return no imaginary frequencies.
3.  Mock `ElasticCalc` to return positive definite elastic tensor ($C_{11} > C_{12}$).
4.  Mock `EOSCalc` to return a smooth energy-volume curve.
5.  Call `validator.validate(potential)`.
6.  Assert that `result.passed` is True.
7.  Assert that `result.metrics['phonon_stable']` is True.

### Scenario 8.2: Detection of Instability
**Priority**: High
**Description**: Verify that the validator correctly flags potentials with physical issues (e.g., imaginary phonon modes indicating structural instability).

**Jupyter Notebook**: `tutorials/07_validation_test.ipynb`
1.  Mock `PhononCalc` to return a large imaginary frequency at the X point.
2.  Call `validator.validate(potential)`.
3.  Assert that `result.passed` is False.
4.  Assert that the failure reason mentions "Phonon instability".

### Scenario 8.3: HTML Report Generation
**Priority**: Medium
**Description**: Verify that the system generates a comprehensive and visually accessible report.

**Jupyter Notebook**: `tutorials/07_validation_test.ipynb`
1.  Create a dummy `ValidationMetrics` object with some sample data.
2.  Create dummy plot files (e.g., empty PNGs).
3.  Call `report_generator.generate_html(metrics, plots)`.
4.  Assert that `validation_report.html` is created.
5.  Inspect the file content to ensure it contains the metrics (e.g., "Bulk Modulus: 150 GPa").

## 2. Behavior Definitions

### Phonon Stability Check
**GIVEN** a phonon band structure calculation
**WHEN** the minimum frequency squared $\omega^2_{min}$ is significantly negative (e.g., $< -0.1$ THz$^2$)
**THEN** the validation status should be set to FAIL
**AND** the report should highlight "Imaginary Modes Detected".

### Elastic Stability Check
**GIVEN** the calculated elastic constants $C_{ij}$
**WHEN** the Born stability criteria are evaluated (e.g., $C_{11} - C_{12} > 0$ for cubic)
**THEN** the validation status should reflect whether the crystal is mechanically stable.

### Equation of State Check
**GIVEN** energy-volume data points
**WHEN** fitted to the Birch-Murnaghan equation
**THEN** the bulk modulus $B_0$ should be positive
**AND** the RMSE of the fit should be small (indicating smooth potential behavior).
