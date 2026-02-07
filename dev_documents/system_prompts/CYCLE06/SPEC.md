# Cycle 06 Specification: Production Readiness & CLI

## 1. Summary

**Goal**: Finalise the system for production deployment. This cycle focuses on polishing the Command Line Interface (CLI), ensuring robust error handling, and conducting comprehensive end-to-end testing with the target scenario ("Fe/Pt on MgO"). It also includes the creation of tutorials and documentation.

**Key Deliverables**:
1.  **`pyacemaker` CLI**: A user-friendly CLI with commands like `init` (create project), `run` (start loop), `status` (show progress), `compute` (single-point calculation).
2.  **End-to-End Test Suite**: Automated tests verifying the complete workflow from structure generation to validation.
3.  **Documentation**: README, Tutorials (Jupyter Notebooks).

## 2. System Architecture

Files in **bold** are the primary focus of this cycle.

```ascii
src/mlip_autopipec/
├── **main.py**               # Enhanced CLI with Typer
├── **__init__.py**
└── utils/
    ├── **cli_utils.py**      # Helpers for CLI (progress bars, formatting)
    └── **config_utils.py**   # Config validation/generation
```

## 3. Design Architecture

### 3.1 CLI Commands (`main.py`)

Using `typer` library:

*   `pyacemaker init`: Creates a new directory with a template `config.yaml`.
*   `pyacemaker run`: Starts the active learning loop.
    *   `--config`: Path to config file.
    *   `--mode`: "mock" or "production".
*   `pyacemaker status`: Reads the `dataset.json` and potential history to display a summary (current iteration, best RMSE, dataset sise).
*   `pyacemaker compute`: Utility to run a single Oracle calculation on a structure file (for debugging).

### 3.2 End-to-End Integration

*   **Scenario**: Fe/Pt deposition on MgO.
*   **Workflow**:
    1.  Train MgO bulk potential.
    2.  Train FePt bulk potential.
    3.  Train Interface potential.
    4.  Run MD deposition.
    5.  Run kMC ordering.

## 4. Implementation Approach

1.  **Enhance CLI**: Add `status` and `compute` commands. Improve error messages.
2.  **Write Tutorials**: Create the Jupyter Notebooks defined in `FINAL_UAT.md`.
3.  **Final Polish**: Ensure logging is consistent, temporary files are cleaned up.

## 5. Test Strategy

### 5.1 System Testing
*   **CLI Usage**: Test `pyacemaker init`, `pyacemaker status` on a fresh directory.
*   **Full Loop**: Run a complete cycle with "Mock" backend but "Real" data structures to ensure no serialisation issues.

### 5.2 User Acceptance Testing (UAT)
*   **Tutorial Execution**: Run the generated notebooks in a fresh environment to ensure they work for a new user.
