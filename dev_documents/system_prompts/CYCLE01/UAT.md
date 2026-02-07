# Cycle 01 UAT: Core Framework & Mocks

## 1. Test Scenarios

### Scenario 01: "The Ghost Run" (End-to-End Mock Execution)
**Priority**: Critical
**Description**: Run the entire pipeline with all components set to "Mock". This verifies the Orchestrator logic, data passing between components, and configuration loading.

**Gherkin Definition**:
```gherkin
GIVEN a configuration file "config_mock.yaml"
AND the "oracle" type is set to "mock"
AND the "trainer" type is set to "mock"
AND the "dynamics" type is set to "mock"
WHEN I execute the command "mlip-pipeline run config_mock.yaml"
THEN the system should initialize without errors
AND the loop should run for the specified "max_cycles"
AND the logs should show "MockOracle computing..."
AND the logs should show "MockTrainer training..."
AND the logs should show "MockDynamics running..."
AND the process should exit with code 0
```

### Scenario 02: "Bad Config" (Validation Error)
**Priority**: High
**Description**: Provide an invalid configuration to ensure Pydantic validation is working.

**Gherkin Definition**:
```gherkin
GIVEN a configuration file "config_bad.yaml"
AND the "oracle" section is missing required fields
WHEN I execute the command "mlip-pipeline run config_bad.yaml"
THEN the system should print a clear validation error message
AND the process should exit with a non-zero code
```

## 2. Verification Steps

1.  **Generate Config**: Create a `config_mock.yaml` using the helper or manually.
2.  **Run Pipeline**: Execute `uv run mlip-pipeline run config_mock.yaml`.
3.  **Check Output**:
    *   Verify `logs/mlip.log` contains entries from all mock components.
    *   Verify dummy artifacts (e.g., `potentials/mock.yace`) are created (even if empty).
