# MLIP-AutoPipe: The Zero-Human Protocol

This repository contains the source code for the Machine Learning Interatomic Potential - Automated Pipeline (MLIP-AutoPipe) project. It is a fully autonomous system designed to eliminate human intervention from the process of generating and validating machine learning interatomic potentials for materials science.

## Cycle 01: Foundation

This initial development cycle focused on establishing the foundational bedrock of the application. The key achievements include:

*   **Project Scaffolding**: The complete directory structure for the source code (`src/mlip_autopipec`) and tests (`tests/`) has been created.
*   **Dependency Management**: The project has been configured with `uv`, and all necessary dependencies for Cycle 01 are defined in `pyproject.toml`.
*   **Schema-Driven Design**: A suite of robust, validated Pydantic models has been implemented in `src/mlip_autopipec/schemas`. These schemas define the core data structures for user input, system configuration, and DFT calculations, ensuring data integrity throughout the pipeline.
*   **Core Utilities**: Foundational utility modules have been created, including a data-driven heuristic engine (`src/mlip_autopipec/utils/qe_utils.py`) for selecting Quantum Espresso parameters.
*   **Testing Framework**: A comprehensive testing suite using `pytest` has been established, with initial unit tests for the Pydantic schemas.
*   **Code Quality**: The project is configured with `ruff` and `mypy` to enforce strict code quality, linting, and static typing standards.
