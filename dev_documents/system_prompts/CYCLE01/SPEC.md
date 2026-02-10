# Cycle 01 Specification: Core Framework & Orchestration

## 1. Summary

This cycle establishes the foundation of the PYACEMAKER system. The primary goal is to create a "Walking Skeleton"—a thin but complete slice of the application that connects the Command Line Interface (CLI), Configuration Parser, Logger, and the central Orchestrator loop.

To allow for rapid development and testing of the logic without waiting for heavy physical simulations, we will implement **Mock Components** for the Structure Generator, Oracle, and Trainer. This ensures that the state transitions of the active learning loop (Exploration -> Labeling -> Training) can be verified purely in software before integrating real physics engines.

## 2. System Architecture

The following file structure will be created. **Bold** files are the targets for this cycle.

```ascii
src/mlip_autopipec/
├── **main.py**                     # CLI Entry Point (typer)
├── **config.py**                   # Global Configuration Loading (pydantic)
├── **constants.py**                # Global Constants
├── core/
│   ├── **orchestrator.py**         # Main Workflow Manager
│   ├── **state_manager.py**        # Persistence (Checkpointing)
│   └── **logger.py**               # Centralized Logging
├── domain_models/
│   ├── **config.py**               # Pydantic Schemas (Config)
│   ├── **datastructures.py**       # Structure, Dataset, Potential
│   └── **enums.py**                # Enums (Status, TaskTypes)
└── components/
    ├── **base.py**                 # Abstract Base Classes
    └── **mock.py**                 # Mock Implementations for Gen/Oracle/Trainer
tests/
├── **test_core.py**                # Tests for Orchestrator & State
└── **test_config.py**              # Tests for Configuration Loading
```

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/`)
We leverage Pydantic V2 for strict type validation.

*   **`Config`**: The root configuration object. It must validate the `config.yaml` file.
    *   `orchestrator`: Settings for work directory, max iterations.
    *   `generator`: Type (random/adaptive), params.
    *   `oracle`: Type (qe/mock), command.
    *   `trainer`: Type (pace/mock), dataset path.
*   **`WorkflowState`**: A JSON-serializable state object to track progress.
    *   `iteration`: int
    *   `current_stage`: Enum (EXPLORE, LABEL, TRAIN)
    *   `latest_potential_path`: Path
*   **`ComponentConfig`**: Base class for all component configs, utilizing `Discriminated Unions` to select between Real and Mock implementations based on a `type` field.

### 3.2. Core Components (`core/`)
*   **`Orchestrator`**: The brain. It holds references to the components and the state.
    *   `run()`: The main loop. checks state, delegates to components, updates state, saves state.
*   **`StateManager`**: Handles loading and saving `workflow_state.json`. It guarantees atomic writes to prevent corruption.

### 3.3. Interfaces (`components/base.py`)
Abstract Base Classes (ABCs) define the contract for all future plugins.
*   `BaseGenerator.generate(n: int) -> List[Structure]`
*   `BaseOracle.compute(structures: List[Structure]) -> List[Structure]`
*   `BaseTrainer.train(dataset: Dataset) -> Potential`

## 4. Implementation Approach

1.  **Domain Setup**: Define `enums.py` (Stage, Status), `datastructures.py` (Structure), and `config.py` (Pydantic models).
2.  **Infrastructure**: Implement `logger.py` using `rich` for pretty console output and file logging.
3.  **Components**: Create `base.py` ABCs and `mock.py`. The mock components should log their actions (e.g., "MockOracle: Computed 5 structures") and return dummy data.
4.  **Orchestrator**: Implement the `run_loop` logic.
    *   Initialize components based on config.
    *   Load or create state.
    *   Loop until `max_iterations`.
    *   Inside loop: Generator -> Oracle -> Trainer.
5.  **CLI**: Use `typer` or `argparse` in `main.py` to expose commands:
    *   `init`: Generate default config.
    *   `run`: Execute the loop.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_config.py`**:
    *   Load a valid YAML and assert fields are parsed correctly.
    *   Load an invalid YAML (missing fields, wrong types) and assert `ValidationError` is raised.
*   **`test_state_manager.py`**:
    *   Save state to a temporary file, load it back, and assert equality.

### 5.2. Integration Testing
*   **`test_core.py`**:
    *   **Scenario**: Run the `Orchestrator` with `Mock` components for 2 iterations.
    *   **Verification**:
        *   Check that `workflow_state.json` shows `iteration: 2`.
        *   Check logs to ensure Generator, Oracle, and Trainer were called in the correct order.
