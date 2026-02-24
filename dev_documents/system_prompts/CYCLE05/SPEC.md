# Cycle 05 Specification: Delta Learning & Full Orchestration

## 1. Summary
Cycle 05 implements the final and most critical step of the MACE Distillation Workflow: **Delta Learning** (Step 7). This step corrects the systematic errors ("Sim-to-Real gap") of the base ACE potential by fine-tuning it on the high-fidelity DFT data collected in Step 2.

Additionally, this cycle focuses on **Full Orchestration**. We will connect all 7 steps into a seamless, automated pipeline. The `Orchestrator` will manage the state transitions, ensuring that the system can recover from interruptions (idempotency) and that data flows correctly from one stage to the next.

## 2. System Architecture

The following file structure will be created or modified. **Bold** files are new or significantly modified.

```text
src/pyacemaker/
├── trainer/
│   └── **pacemaker.py**      # Updated: Delta Learning Logic (support for initial_potential)
├── domain_models/
│   └── **state.py**          # New: Pipeline State Persistence
├── modules/
│   └── **mace_workflow.py**  # Updated: Expose steps for Orchestrator
└── **orchestrator.py**       # Updated: Full 7-Step Logic with State Machine
```

## 3. Design Architecture

### 3.1. Delta Learning (`trainer/pacemaker.py`)
-   **`PacemakerTrainer.train(dataset, initial_potential=..., weight_dft=...)`**:
    -   Enhance existing `train` method to support Delta Learning configuration.
    -   If `initial_potential` is provided, it should be used as the starting point (baseline or load-potential).
    -   If `weight_dft` is provided (via kwargs), it should set `w_energy` and `w_forces` to emphasize DFT data.
    -   **Output**: A `final_potential.yace` file.

### 3.2. Orchestration State (`domain_models/state.py`)
-   **`PipelineState`**: A Pydantic model saved to `pipeline_state.json`.
    -   `current_step`: int (1-7).
    -   `completed_steps`: List[int].
    -   `artifacts`: Dict[str, Path] (e.g., {"pool_path": "...", "fine_tuned_potential": "...", "surrogate_dataset": "..."}).

### 3.3. Full Pipeline Logic (`orchestrator.py`)
-   **`Orchestrator.run()`**:
    -   Checks `pipeline_state.json`.
    -   Resumes from the last incomplete step.
    -   Sequentially executes steps exposed by `MaceDistillationWorkflow`.
    -   Updates state after each step.

## 4. Implementation Approach

1.  **Implement State Management**: Create `PipelineState` in `domain_models/state.py`.
2.  **Refactor Workflow**: Update `MaceDistillationWorkflow` in `modules/mace_workflow.py` to expose individual steps (`step1_direct_sampling`, etc.) and return necessary artifacts.
3.  **Implement Delta Learning**: Update `PacemakerTrainer` in `trainer/pacemaker.py` to support `initial_potential` in `_generate_input_yaml`.
4.  **Finalize Orchestrator**: Update `orchestrator.py` to implement the full state machine loop.

## 5. Test Strategy

### 5.1. Unit Testing
-   **State Persistence**: Verify that `PipelineState` correctly saves and loads from JSON.
-   **Delta Configuration**: Verify that `PacemakerTrainer` generates the correct command-line arguments or config for fine-tuning.

### 5.2. Integration Testing
-   **Full Pipeline (Mock Mode)**:
    1.  Start a fresh run with `config.yaml`.
    2.  Interrupt the process (simulate crash) after Step 3.
    3.  Restart the process.
    4.  Verify that it skips Steps 1-3 and resumes at Step 4.
    5.  Verify that it completes all 7 steps and produces `final_potential.yace`.
