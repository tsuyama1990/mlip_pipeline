# Cycle 01 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 01-01: Valid Configuration Loading
- **Priority**: Critical
- **Description**: Ensure the system correctly parses a comprehensive `config.yaml` file.
- **Pre-conditions**: A valid `config.yaml` exists.
- **Steps**:
  1. Run `mlip-auto init --dry-run config.yaml`
- **Expected Result**: System prints "Configuration Valid" and dumps the parsed structure to stdout.

### Scenario 01-02: Invalid Configuration Rejection
- **Priority**: Critical
- **Description**: Ensure the system rejects invalid configs (wrong types, missing keys).
- **Steps**:
  1. Create `bad_config.yaml` with string where int is expected (e.g., `temperature: "high"`).
  2. Run `mlip-auto init --dry-run bad_config.yaml`
- **Expected Result**: System exits with non-zero code and prints a clear, helpful Pydantic error message indicating the specific field.

### Scenario 01-03: Database Initialization & Persistence
- **Priority**: High
- **Description**: Verify the database is created and stores data persistence.
- **Steps**:
  1. Use a python script to instantiate `DatabaseManager`.
  2. Add an ASE Atoms object.
  3. Close connection.
  4. Re-open connection and count rows.
- **Expected Result**: Row count is 1. Metadata matches input.

## 2. Behavior Definitions

```gherkin
Feature: Configuration Management

  Scenario: Load valid configuration
    GIVEN a configuration file "valid_config.yaml"
    WHEN the user runs "mlip-auto validate valid_config.yaml"
    THEN the exit code should be 0
    AND the output should contain "Validation Successful"

  Scenario: Fail on missing required fields
    GIVEN a configuration file "missing_field.yaml"
    WHEN the user runs "mlip-auto validate missing_field.yaml"
    THEN the exit code should be 1
    AND the output should contain "Field required"
```
