# Cycle 01 UAT: System Skeleton & Mock Infrastructure

## 1. Test Scenarios

### Scenario 01: "The Hello World of Active Learning"
**Priority**: Critical
**Description**: Verify that the system can initialize and run a complete loop using mock components. This ensures the plumbing (config, logging, orchestration) is leak-proof.

**Pre-conditions**:
-   Python 3.12+ environment with dependencies installed.
-   No existing output directories (`mlip_run/`).

**Steps**:
1.  User runs `pyacemaker init` to generate a default `config.yaml`.
2.  User edits `config.yaml` to ensure `type: mock` is set for all components (Oracle, Trainer, etc.).
3.  User runs `pyacemaker run --config config.yaml`.

**Expected Outcome**:
-   The command finishes within 5 seconds.
-   A `logs/` directory is created with a log file showing "Cycle 1 started", "Cycle 1 finished".
-   A `potentials/` directory contains `generation_000.yace`.
-   No python tracebacks are printed to the console.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: System Initialization and Mock Execution

  Scenario: Run full cycle with Mock components
    Given I have a configuration file "config.yaml"
    And the configuration sets "oracle.type" to "mock"
    And the configuration sets "trainer.type" to "mock"
    And the configuration sets "dynamics.type" to "mock"
    When I execute the command "pyacemaker run --config config.yaml"
    Then the process should exit with status code 0
    And the output directory should contain "potential.yace"
    And the log file should contain "Orchestrator: Cycle 1 completed"
```
