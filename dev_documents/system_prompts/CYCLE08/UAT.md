# Cycle 08 UAT: Validation Suite

## 1. Test Scenarios

| ID | Scenario Name | Description | Priority |
| :--- | :--- | :--- | :--- |
| **08-1** | **Phonon Stability Check** | Verify that the system correctly identifies stable (all real freq) and unstable (imaginary freq) crystals. | High |
| **08-2** | **Elastic Constants Accuracy** | Verify that calculated elastic moduli match reference values (within tolerance) for standard materials. | Critical |
| **08-3** | **HTML Report Generation** | Verify that a comprehensive HTML report (with plots) is generated after the validation phase. | Medium |

## 2. Behavior Definitions (Gherkin)

### Scenario 08-1: Phonon Stability Check
```gherkin
GIVEN a potential for "Stable FCC Al"
WHEN the Validator runs the Phonon analysis
THEN the maximum imaginary frequency should be 0.0 THz (or negligible)
AND a "phonon_dispersion.png" plot should be saved
```

### Scenario 08-3: HTML Report Generation
```gherkin
GIVEN a completed Validation phase with EOS, Elastic, and Phonon results
WHEN the Report Generator is called
THEN a "validation_report.html" file should be created
AND the file should contain a table with "Bulk Modulus"
AND the file should embed the "phonon_dispersion.png" image
```
