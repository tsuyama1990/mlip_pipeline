# Cycle 05 Specification: Delta Learning & Full Orchestration

## 1. Summary
Cycle 05 implements the final and most critical step of the MACE Distillation Workflow: **Delta Learning** (Step 7). This step corrects the systematic errors ("Sim-to-Real gap") of the base ACE potential by fine-tuning it on the high-fidelity DFT data collected in Step 2.

Additionally, this cycle focuses on **Full Orchestration**. We will connect all 7 steps into a seamless, automated pipeline. The `Orchestrator` will manage the state transitions, ensuring that the system can recover from interruptions (idempotency) and that data flows correctly from one stage to the next.

## 2. System Architecture

The following file structure will be created or modified. **Bold** files are new or significantly modified.

```text
src/pyacemaker/
├── trainer/
│   └── **pacemaker.py**      # Updated: Delta Learning Logic
├── core/
│   └── **state.py**          # Pipeline State Persistence
└── **orchestrator.py**       # Updated: Full 7-Step Logic
```

## 3. Design Architecture

### 3.1. Delta Learning (`trainer/pacemaker.py`)
-   **`PacemakerTrainer.train_delta(base_potential: Path, dft_dataset: Path, weight: float)`**:
    -   Loads the pre-trained base ACE potential (`potential.yace`).
    -   Configures Pacemaker to use the `dft_dataset` for fine-tuning.
    -   **Key Mechanism**: Uses a high weight for the DFT data (e.g., 10x or 100x relative to the base training weight) to force the potential to prioritize DFT accuracy while maintaining the general shape learned from MACE.
    -   **Output**: A `final_potential.yace` file.

### 3.2. Orchestration State (`core/state.py`)
-   **`PipelineState`**: A Pydantic model saved to `pipeline_state.json`.
    -   `current_step`: int (1-7).
    -   `completed_steps`: List[int].
    -   `artifacts`: Dict[str, Path] (e.g., {"dft_dataset": "path/to/dft.pckl", "surrogate_model": "path/to/mace.model"}).

### 3.3. Full Pipeline Logic (`orchestrator.py`)
-   **`Orchestrator.run()`**:
    -   Checks `pipeline_state.json`.
    -   Resumes from the last incomplete step.
    -   Sequentially executes `run_step1` through `run_step7`.
    -   Updates state after each step.

## 4. Implementation Approach

1.  **Implement Delta Learning**: Update `PacemakerTrainer` in `trainer/pacemaker.py` to support "restart" training or weighted dataset mixing.
2.  **Implement State Management**: Create `PipelineState` in `core/state.py`.
3.  **Finalize Orchestrator**: Update `orchestrator.py` to implement the full state machine and error handling.
    -   Wrap each step in a try-except block to save state on failure.

## 5. Test Strategy

### 5.1. Unit Testing
-   **State Persistence**: Verify that `PipelineState` correctly saves and loads from JSON.
-   **Delta Configuration**: Verify that `PacemakerTrainer` generates the correct command-line arguments or config for fine-tuning (e.g., `--load-potential=potential.yace`).

### 5.2. Integration Testing
-   **Full Pipeline (Mock Mode)**:
    1.  Start a fresh run with `config.yaml`.
    2.  Interrupt the process (simulate crash) after Step 3.
    3.  Restart the process.
    4.  Verify that it skips Steps 1-3 and resumes at Step 4.
    5.  Verify that it completes all 7 steps and produces `final_potential.yace`.
    6.  Verify that `final_potential.yace` exists and is distinct from the base `potential.yace`.
