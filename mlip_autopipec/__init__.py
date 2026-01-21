"""
MLIP-AutoPipe Package.

Architecture:
-------------
MLIP-AutoPipe follows a modular architecture orchestrated by a central WorkflowManager.

Components:
1.  **Config (`mlip_autopipec.config`)**: Pydantic schemas acting as the contract for all modules.
    -   `MLIPConfig` is the root configuration.
    -   `SystemConfig` is the comprehensive internal state.

2.  **Core (`mlip_autopipec.core`)**: Shared infrastructure.
    -   `DatabaseManager`: Wraps `ase.db` (SQLite) for storing atomic structures and metadata.
    -   `Logging`: Centralized logging configuration using `rich`.

3.  **Orchestration (`mlip_autopipec.orchestration`)**:
    -   `TaskQueue`: Manages distributed task execution using `dask.distributed`.
    -   `WorkflowManager`: Implements the active learning state machine.

4.  **Execution Modules**:
    -   `Generator`: Creates initial structures.
    -   `Surrogate`: Filters structures.
    -   `DFT`: Runs Quantum Espresso.
    -   `Training`: Trains potentials (Pacemaker).
    -   `Inference`: Runs MD (LAMMPS).

Interactions:
-   The `WorkflowManager` reads `SystemConfig` and initializes the `DatabaseManager` and `TaskQueue`.
-   It queries the DB for tasks.
-   It submits tasks (e.g., DFT calculations) to the `TaskQueue`.
-   `TaskQueue` executes functions (e.g., `QERunner.run`) on workers.
-   Results are written back to the DB via `DatabaseManager`.

Dependency Management:
-   Managed via `uv`. `pyproject.toml` defines strict version constraints.
"""
