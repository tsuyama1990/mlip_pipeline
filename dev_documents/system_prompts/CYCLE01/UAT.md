# Cycle 01: User Acceptance Test (UAT) Plan

## 1. Test Scenarios

### Scenario 1.1: Configuration Loading
**Priority**: Critical
**Description**: Verify that the system can correctly parse a valid configuration file and reject an invalid one. This is the foundation for all subsequent operations.

**Jupyter Notebook**: `tutorials/00_config_test.ipynb` (Create this notebook for testing)
1.  Create a dummy `config.yaml` with valid settings for all components.
2.  Import `ExperimentConfig` from `mlip_autopipec.config`.
3.  Attempt to load the file using `ExperimentConfig.from_yaml("config.yaml")`.
4.  Assert that the returned object is an instance of `ExperimentConfig`.
5.  Access nested fields (e.g., `config.oracle.type`) and verify values match the YAML.

### Scenario 1.2: Invalid Configuration Handling
**Priority**: High
**Description**: Verify that the system provides helpful error messages when configuration is incorrect.

**Jupyter Notebook**: `tutorials/00_config_test.ipynb`
1.  Create a `bad_config.yaml` with a missing required field (e.g., `oracle.type`).
2.  Attempt to load it inside a `try...except` block.
3.  Assert that a `pydantic.ValidationError` is raised.
4.  Verify that the error message clearly identifies the missing field.

### Scenario 1.3: Component Instantiation via Factory
**Priority**: High
**Description**: Verify that the Orchestrator can instantiate dummy components using the Factory pattern.

**Jupyter Notebook**: `tutorials/00_config_test.ipynb`
1.  Define a dummy `MockOracle` class and register it with the Factory.
2.  Create a config specifying `type: "MOCK"`.
3.  Instantiate the `Orchestrator` with this config.
4.  Verify that `orchestrator.oracle` is an instance of `MockOracle`.

## 2. Behavior Definitions

### Configuration Validation
**GIVEN** a configuration file with `oracle.type = "QE"`
**AND** `oracle.pseudopotentials` is missing
**WHEN** the configuration is loaded
**THEN** the system should raise a `ValidationError`
**AND** the error message should mention "pseudopotentials".

### Factory Instantiation
**GIVEN** a configuration with `generator.type = "RANDOM"`
**WHEN** the `ComponentFactory` is called to create the generator
**THEN** it should return an instance of `RandomGenerator` (or Mock equivalent in this cycle)
**AND** the instance should have the correct parameters from the config.
