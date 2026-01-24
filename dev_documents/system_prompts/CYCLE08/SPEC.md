# Cycle 08 Specification: Orchestration & CLI (The Final Integration)

## 1. Summary

Cycle 08 is the culmination of the project. We have built the engine parts (Generator, Surrogate, DFT, Trainer, Inference); now we build the chassis and the steering wheel.

We implements the **Workflow Manager**, the central brain that orchestrates the cyclical flow of data. It manages the state of the active learning loop, deciding when to generate data, when to run DFT, when to train, and when to run inference. It ensures that the pipeline is robust, resumable, and observable.

We also implement the **CLI (Command Line Interface)**, the primary interface for the user. Using `typer`, we provide a suite of commands (`init`, `generate`, `run-loop`, `status`) that allow users to interact with the complex machinery using simple keywords.

By the end of this cycle, `mlip-auto run-loop input.yaml` will trigger the full "Zero-Human" protocol, running for days or weeks until the material properties are converged.

## 2. System Architecture

Files marked in **bold** are new or modified in this cycle.

### 2.1. File Structure

```ascii
mlip_autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── **app.py**                  # The Typer Application
│       ├── config/
│       │   └── schemas/
│       │       └── **workflow.py**     # Workflow Configuration
│       └── **orchestration/**
│           ├── **__init__.py**
│           ├── **workflow.py**         # The Main Loop Logic
│           └── **state.py**            # State Persistence
└── tests/
    └── orchestration/
        ├── **test_workflow.py**
        └── **test_app.py**
```

### 2.2. Code Blueprints

#### `src/mlip_autopipec/orchestration/state.py`
Tracks the progress.

```python
from pydantic import BaseModel
from typing import List

class WorkflowState(BaseModel):
    iteration: int = 0
    current_phase: str = "IDLE" # GENERATION, DFT, TRAINING, INFERENCE
    potential_versions: List[str] = []

    def save(self, path: str):
        with open(path, 'w') as f:
            f.write(self.model_dump_json())

    @classmethod
    def load(cls, path: str):
        # ...
        pass
```

#### `src/mlip_autopipec/orchestration/workflow.py`
The conductor.

```python
from mlip_autopipec.config.models import MLIPConfig
from mlip_autopipec.dft.runner import QERunner
from mlip_autopipec.training.pacemaker import PacemakerWrapper
# ... imports

class WorkflowManager:
    def __init__(self, config: MLIPConfig):
        self.config = config
        self.state = WorkflowState.load("state.json")
        # Initialize sub-managers

    def run_loop(self):
        while self.state.iteration < self.config.workflow.max_iterations:
            if self.state.current_phase == "GENERATION":
                self.run_generation_phase()
            elif self.state.current_phase == "DFT":
                self.run_dft_phase()
            elif self.state.current_phase == "TRAINING":
                self.run_training_phase()
            elif self.state.current_phase == "INFERENCE":
                self.run_inference_phase()

            self.state.iteration += 1
            self.state.save("state.json")
```

#### `src/mlip_autopipec/app.py`
The facade.

```python
import typer
from mlip_autopipec.orchestration.workflow import WorkflowManager

app = typer.Typer()

@app.command()
def loop(config_path: str = "input.yaml"):
    """Starts the autonomous active learning loop."""
    config = load_config(config_path)
    manager = WorkflowManager(config)
    manager.run_loop()

if __name__ == "__main__":
    app()
```

## 3. Design Architecture

### 3.1. Domain Concepts

1.  **Resumability**: The pipeline must be crash-proof. If the server reboots during "Step 50 of DFT", the `WorkflowManager` must check the DB on restart, see that 50 jobs are done and 50 are pending, and resume processing. We rely on the `state.json` and the Database status columns for this.
2.  **Phase Transition Gates**:
    -   *Generation -> DFT*: Gate is "Are candidates generated?"
    -   *DFT -> Training*: Gate is "Are there sufficient new COMPLETED structures?"
    -   *Training -> Inference*: Gate is "Is potential.yace created?"
    -   *Inference -> DFT*: Gate is "Did we find high-uncertainty structures?"

### 3.2. Consumers and Producers

-   **Consumer**: `WorkflowManager` consumes the entire `MLIPConfig`.
-   **Producer**: It produces the final result (the converged potential and property reports).

## 4. Implementation Approach

### Step 1: State Management
-   **Task**: Implement `WorkflowState`.
-   **Detail**: Simple JSON persistence. Must handle atomic writes to avoid corruption.

### Step 2: The Loop Logic
-   **Task**: Implement `WorkflowManager`.
-   **Detail**: This is the "God Class" that instantiates all other classes. However, to keep it clean, it should delegate heavily.
    -   `run_dft_phase()` -> loops over DB pending items -> calls `QERunner`.

### Step 3: CLI Polish
-   **Task**: Update `app.py`.
-   **Detail**: Add nice progress bars (`tqdm`) for long phases. Add `verbose` flags.

## 5. Test Strategy

### 5.1. Unit Testing Approach (Min 300 words)

-   **State Transitions**:
    -   *Test*: Initialize Manager in "DFT" phase.
    -   *Action*: Mock `QERunner` to complete all jobs immediately.
    -   *Assert*: State transitions to "TRAINING".
-   **Gate Logic**:
    -   *Test*: Manager in "DFT" phase, but DB is empty.
    -   *Assert*: Logs warning "Nothing to calculate", transitions to next appropriate phase or exits.

### 5.2. Integration Testing Approach (Min 300 words)

-   **The "Dry Run" (Grand Unified Test)**:
    -   This is the most important test of the project.
    -   *Setup*: Mocks for QE (returns dummy energy), Pacemaker (returns dummy potential), LAMMPS (returns dummy uncertainty).
    -   *Config*: `max_iterations: 2`.
    -   *Execution*: Run `mlip-auto loop`.
    -   *Verification*:
        1.  Cycle 1 starts.
        2.  Gen phase calls Builder.
        3.  DFT phase "processes" structures.
        4.  Train phase "creates" potential v1.
        5.  Inf phase "finds" uncertainty.
        6.  Cycle 2 starts.
        7.  DFT phase processes new candidates.
        8.  Train phase creates potential v2.
        9.  Loop exits.
    -   *Result*: `state.json` shows `iteration: 2`. DB has entries. Files exist.
