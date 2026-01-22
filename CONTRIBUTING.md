# Contributing to MLIP-AutoPipe

We welcome contributions! Please follow these guidelines.

## Development Setup

1.  Clone the repository.
2.  Install `uv` (https://astral.sh/uv).
3.  Run `uv sync` to install dependencies.
4.  Run `uv run pre-commit install` (if configured) or ensure you run linting manually.

## Standards

-   **Linting**: We use `ruff` and `mypy`. Run `uv run ruff check .` and `uv run mypy .` before submitting.
-   **Testing**: We practice TDD. Write tests in `tests/` before implementing features. Ensure 85% coverage.
-   **Code Style**: Follow PEP 8 (enforced by `ruff`).
-   **Type Hints**: All functions must have type hints.

## Pull Requests

1.  Open a PR with a descriptive title.
2.  Ensure all tests pass.
3.  Request review.
