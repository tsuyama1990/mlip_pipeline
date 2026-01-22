# Testing Strategy

## Overview
We aim for high reliability and maintainability. Testing is split into Unit, Integration, and UAT layers.

## Mocking Guidelines

### 1. What to Mock
-   **External Binaries**: Always mock calls to external heavy executables like `pacemaker`, `lmp` (LAMMPS), and `pw.x` (Quantum Espresso). We do not want to run actual DFT or MD in unit/integration tests.
    -   *Technique*: Use `unittest.mock.patch` on `subprocess.run` or the specific Runner class methods.
-   **Filesystem I/O**: For unit tests, use `tmp_path` fixture or `pyfakefs`. For integration, use isolated `tmp_path` directories.
-   **External Services**: Dask schedulers should be mocked or used in `LocalCluster` mode for integration.

### 2. What NOT to Mock
-   **Internal Logic**: Do not mock the class under test.
-   **Data Models**: Use real Pydantic models (e.g., `TrainingConfig`, `InferenceConfig`) instead of mocks to ensure validation logic is tested.
-   **Database (Integration)**: Use a real SQLite file in a temporary directory (`tmp_path / "test.db"`) via `DatabaseManager`. Do NOT mock `ase.db` in integration tests, as we need to verify query logic.

### 3. Isolation
-   Every test should run in a clean state.
-   Use `pytest` fixtures for setup/teardown.
-   Do not rely on global state.

## Test Categories

### Unit Tests (`tests/unit/`)
-   Focus: Individual functions and classes.
-   Speed: Fast (< 1s).
-   Mocks: Heavy use of mocks for dependencies.

### Integration Tests (`tests/integration/`)
-   Focus: Interaction between modules (e.g., `WorkflowManager` -> `DatabaseManager`).
-   Speed: Moderate.
-   Persistence: Use temporary file-based SQLite databases.

### User Acceptance Tests (UAT) (`tests/uat/`)
-   Focus: End-to-end user workflows via CLI.
-   Tooling: `typer.testing.CliRunner`.
-   Verification: Check exit codes and stdout.
