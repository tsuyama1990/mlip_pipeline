# Cycle 02 UAT: Oracle & Data Management

## 1. Test Scenarios

### Scenario 01: Dataset I/O Verification
**Priority**: High
**Description**: Verify that `DatasetManager` can correctly serialize and deserialize a list of `ase.Atoms` objects using `pickle` and `gzip`.
**Steps**:
1.  Create a python script `test_dataset.py`.
2.  Instantiate a list of 10 random `Atoms` objects.
3.  Save them to `data/test.pckl.gzip` using `DatasetManager.save`.
4.  Load them back using `DatasetManager.load`.
5.  Assert that the length is 10 and properties (positions, symbols) match.
**Expected Result**:
-   Exit code 0.
-   Data matches exactly.

### Scenario 02: Mock DFT Execution
**Priority**: Critical
**Description**: Verify that `DFTManager` correctly configures the ASE calculator and "runs" a calculation (mocked) without error.
**Steps**:
1.  Create a python script `test_dft_mock.py`.
2.  Define a `DFTConfig` with dummy paths.
3.  Mock `ase.calculators.espresso.Espresso.get_potential_energy` to return -100.0 eV.
4.  Run `DFTManager.compute(structure)`.
**Expected Result**:
-   The structure object now has `calc` attached.
-   `structure.get_potential_energy()` returns -100.0.

### Scenario 03: Self-Healing Logic (Retry Mechanism)
**Priority**: Medium
**Description**: Verify that `DFTManager` attempts to retry a calculation with different parameters upon failure.
**Steps**:
1.  Create a python script `test_retry.py`.
2.  Mock `get_potential_energy` to raise `Exception("SCF not converged")` on the first call, and succeed on the second.
3.  Spy on the `_create_calculator` method to inspect arguments.
4.  Run `DFTManager.compute(structure)`.
**Expected Result**:
-   The function succeeds eventually.
-   The spy shows that `_create_calculator` was called twice.
-   The second call had different parameters (e.g., lower `mixing_beta`).

### Scenario 04: Config Validation
**Priority**: Low
**Description**: Verify that the system checks for the existence of pseudopotential files before running.
**Steps**:
1.  Create a `config.yaml` pointing to non-existent pseudopotentials.
2.  Attempt to initialize `DFTManager`.
**Expected Result**:
-   `ConfigurationError` or `FileNotFoundError` is raised immediately.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Oracle & Data Management

  Scenario: Save and Load Dataset
    Given a list of 10 atomic structures
    When I save them to "data/test.pckl.gzip"
    And I load them back from "data/test.pckl.gzip"
    Then the loaded list should contain 10 structures
    And the structures should be identical to the original list

  Scenario: Run DFT Calculation (Mock)
    Given a valid structure and DFT configuration
    When I request a DFT calculation
    Then the system should configure the ASE calculator
    And the structure should have potential energy attached

  Scenario: Retry failed DFT Calculation
    Given a structure that causes SCF convergence failure on the first try
    When I request a DFT calculation
    Then the system should catch the error
    And the system should retry with modified parameters (e.g., lower mixing_beta)
    And the calculation should eventually succeed (if mocked to succeed)
```
