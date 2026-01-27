# Cycle 01 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 01-01: Valid Configuration Loading
- **Priority**: High
- **Description**: Verify that the system correctly loads a valid `config.yaml` and initializes the internal state.
- **Steps**:
    1. Create a `config.yaml` with valid paths for `pseudo_dir` and valid settings for `dft`.
    2. Run `mlip-auto validate --config config.yaml`.
    3. **Expected Result**: System prints "Configuration Valid" and exits with code 0.

### Scenario 01-02: DFT Execution with Silicon
- **Priority**: Critical
- **Description**: Verify that the Oracle module can take a structure, run QE, and return results.
- **Steps**:
    1. Prepare a `config.yaml` pointing to a valid QE executable (or mock).
    2. Use a Python script (Jupyter Notebook recommended) to instantiate `QERunner`.
    3. Pass a 2-atom Silicon primitive cell.
    4. Call `runner.run(atoms)`.
    5. **Expected Result**: A `DFTResult` object is returned containing Energy (float), Forces (Nx3 array), and Stress (3x3 array).

### Scenario 01-03: Self-Healing Logic
- **Priority**: Medium
- **Description**: Verify that the system retries calculation upon convergence failure.
- **Steps**:
    1. Mock the QE executable to return a "convergence not achieved" error on the first run, and success on the second run.
    2. Run the `QERunner`.
    3. **Expected Result**: The logs show "Convergence failed. Retrying with reduced mixing beta...". The final result is successful.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Core Configuration and Oracle

  Scenario: Loading valid configuration
    GIVEN a configuration file "config.yaml" exists
    AND the field "dft.command" is "pw.x"
    WHEN I run the command "mlip-auto init"
    THEN the system should initialize without errors

  Scenario: Running DFT on Silicon
    GIVEN the QERunner is initialized
    AND I have a Silicon crystal structure
    WHEN I request a static calculation
    THEN the system should generate a "pw.in" file
    AND the input file should contain "tprnfor = .true."
    AND the system should return the potential energy
```
