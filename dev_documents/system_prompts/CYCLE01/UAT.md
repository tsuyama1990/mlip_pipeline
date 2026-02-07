# Cycle 01 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1.1: Config Parsing & Structure Validation
**Goal**: Verify that the system correctly parses configuration files and validates data structures.
**Priority**: High (P0) - Foundation must be solid.
**Steps**:
1.  Create a `config.yaml` with `type: mock` for all components.
2.  Run a script that loads this config into `GlobalConfig`.
3.  Create a `Structure` object with valid and invalid arrays (e.g., positions shape (N, 2) instead of (N, 3)).
**Success Criteria**:
*   Valid config loads without error.
*   Invalid structure raises `pydantic.ValidationError`.

### Scenario 1.2: The "Mock Loop" Execution
**Goal**: Verify that all components (MockOracle, MockTrainer, etc.) can communicate via the defined interfaces.
**Priority**: High (P0) - Data flow verification.
**Steps**:
1.  Initialise all Mock components.
2.  Pass a dummy structure to `MockDynamics.run()`.
3.  Assert it returns an `ExplorationResult` (halted or converged).
4.  If halted, pass the structure to `MockGenerator.generate()`.
5.  Pass candidates to `MockOracle.compute()`. Assert energy/forces are added.
6.  Pass labeled data to `MockTrainer.train()`. Assert a potential file is created.
**Success Criteria**:
*   The entire chain executes without `AttributeError` or `TypeError`.
*   Data types flow correctly (Structure -> List[Structure] -> Dataset -> Potential).

## 2. Behaviour Definitions (Gherkin)

```gherkin
Feature: Foundation & Mocks

  Scenario: Load Configuration
    Given a valid "config.yaml" with "mock" backend
    When I load it into GlobalConfig
    Then the oracle config type should be "mock"
    And the workdir should be a valid Path object

  Scenario: Mock Oracle Computation
    Given a list of 5 atomic structures
    And a initialised MockOracle
    When I call compute() on the list
    Then each structure should have "energy" in properties
    And each structure should have "forces" array of shape (N, 3)

  Scenario: Mock Training
    Given a list of labeled structures
    And a initialised MockTrainer
    When I call train()
    Then a file "dummy.yace" should be created in the workdir
    And it should return a Potential object pointing to that file
```
