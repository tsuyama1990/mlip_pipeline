# Contributing to MLIP-AutoPipe

We love your input! We want to make contributing to this project as easy and transparent as possible.

## Pull Request Process

1.  **Fork** the repo and create your branch from `main`.
2.  **Install** dependencies using `uv sync` or `pip install -e .[dev]`.
3.  **Test**: Ensure the test suite passes (`pytest`).
4.  **Lint**: Ensure code follows standards (`ruff check .`, `mypy .`).
5.  **Submit** a Pull Request.

## Coding Standards

-   **Type Hints**: All function arguments and return values must be typed.
-   **Docstrings**: Use Google-style docstrings for all classes and public methods.
-   **Validation**: Use Pydantic schemas for data structures.
-   **Testing**: Add unit tests for new features. Mock external binaries.

## Issue Reporting

-   Use the GitHub Issues tracker.
-   Describe the issue clearly, including steps to reproduce.
