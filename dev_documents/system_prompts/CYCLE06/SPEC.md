# Cycle 06 Specification: Orchestrator & Validation

## 1. Summary
This final cycle integrates all previous components into the cohesive "Orchestrator". We implement the central logic that drives the Active Learning Loop: alternating between Exploration (Dynamics), Labelling (Oracle), Training (Trainer), and Verification (Validator). Additionally, we implement the `Validator` module to perform rigorous physical checks (Phonons, Elasticity) before deploying a potential. Finally, we polish the CLI entry point to expose the full functionality to the user.

## 2. System Architecture

The following file structure will be created/modified. Files in **bold** are the primary deliverables for this cycle.

```ascii
src/
└── mlip_autopipec/
    ├── implementations/
    │   ├── **simple_orchestrator.py** # Main Loop Logic
    │   └── **validator/**
    │       ├── **__init__.py**
    │       ├── **physics_checks.py**  # Phonon/Elastic Logic
    │       └── **report_generator.py** # HTML/YAML Report
    └── **main.py**                    # Final CLI Integration
```

## 3. Design Architecture

### 3.1. SimpleOrchestrator
The `SimpleOrchestrator` implements `BaseOrchestrator`.
-   **State Machine**: Manages the transitions.
    -   `INIT` -> `EXPLORE`
    -   `EXPLORE` -> `LABEL` (if Halted)
    -   `LABEL` -> `TRAIN`
    -   `TRAIN` -> `VALIDATE`
    -   `VALIDATE` -> `DEPLOY` (if Pass) or `EXPLORE` (if Fail/Continue)
-   **Loop Control**: It runs for a maximum number of cycles or until convergence (no halts for $N$ steps).

### 3.2. Validator
The `Validator` class runs a suite of tests on a potential.
-   **Phonons**: Uses `phonopy` (if available) or a simplified finite-displacement method to check for imaginary frequencies (instability).
-   **Elasticity**: Calculates elastic constants ($C_{11}, C_{12}, ...$) and checks Born stability criteria.
-   **Reporting**: Aggregates results into `validation_report.yaml`.

## 4. Implementation Approach

### Step 1: Validator Implementation
Implement `physics_checks.py`.
-   `check_phonons(potential, structure)`: Return boolean (stable/unstable).
-   `check_elasticity(potential, structure)`: Return boolean.

### Step 2: Orchestrator Implementation
Implement `SimpleOrchestrator.run()`.
-   Load initial potential (or train from scratch).
-   **While** `cycle < max_cycles`:
    1.  **Dynamics**: Run MD. If converged, break.
    2.  **Oracle**: If halted, extract structures, run DFT.
    3.  **Trainer**: Update dataset, retrain potential.
    4.  **Validator**: Check new potential.
    5.  **Deploy**: Update the `current.yace` link.

### Step 3: CLI Polish
Update `main.py`.
-   Ensure all arguments are correctly parsed.
-   Add `--dry-run` flag (using Mocks).
-   Add `--debug` flag for verbose logging.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Validator**: Feed a known unstable potential (e.g., negative modulus) and assert that `check_elasticity` returns False.
-   **Orchestrator State**: Mock all components. Verify that the Orchestrator calls them in the correct order (Dynamics -> Oracle -> Trainer).

### 5.2. System Testing (End-to-End)
-   **Mock Run**: Execute `mlip-pipeline run config.yaml` with `type: mock` for all components.
-   Verify that the loop runs for the configured number of cycles.
-   Verify that "dataset.pckl" grows in size (simulated).
-   Verify that "potential_cycle_XX.yace" files are created.
