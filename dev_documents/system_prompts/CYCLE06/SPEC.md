# Cycle 06 Specification: Active Learning Orchestration (MVP)

## 1. Summary
This cycle is the "Integration Phase" where we wire together the previously built components (Explorer, Oracle, Trainer, Validator) into a cohesive **Active Learning Loop**. The Orchestrator will now manage the full lifecycle of data generation, model training, and refinement. We also implement state management (checkpointing) to ensure the long-running process can be paused and resumed. At the end of this cycle, the system will be able to perform a complete "Closed Loop" execution (using mocks or real tools), demonstrating the autonomous improvement of the potential.

## 2. System Architecture

### 2.1. File Structure

```
mlip-pipeline/
├── src/
│   └── mlip_autopipec/
│       ├── orchestration/
│       │   ├── **orchestrator.py**     # [MODIFY] Implement run_loop() logic
│       │   └── **state.py**            # [NEW] State persistence logic
└── tests/
    └── e2e/
        └── **test_loop.py**            # [NEW] Full loop integration tests
```

## 3. Design Architecture

### 3.1. `LoopState` (Pydantic)
Tracks the progress of the workflow.
*   `current_cycle`: int.
*   `total_cycles`: int.
*   `current_potential_path`: Path.
*   `dataset_stats`: Dict (size, composition).
*   `status`: str (RUNNING, PAUSED, COMPLETED, ERROR).

### 3.2. `Orchestrator` Logic
The `run()` method now implements the following sequence:
1.  **Load State**: Check if a checkpoint exists (`state.json`). If so, resume.
2.  **Loop** (`while cycle < max_cycles`):
    *   **Phase 1: Explore**: Call `explorer.explore(current_potential)`. Receive `candidates`.
    *   **Phase 2: Label**: Call `oracle.label(candidates)`. Receive `new_data`.
    *   **Phase 3: Train**: Call `trainer.train(dataset + new_data)`. Receive `new_potential`.
    *   **Phase 4: Validate**: Call `validator.validate(new_potential)`.
    *   **Decision**: If Valid -> Update `current_potential`, Increment Cycle. If Invalid -> Log warning (or implementing retry logic in Cycle 7).
    *   **Checkpoint**: Save `LoopState` to disk.

### 3.3. Checkpointing Strategy
We save `state.json` at the end of each phase. This allows us to recover if the process is killed (e.g., by a cluster scheduler time limit).

## 4. Implementation Approach

1.  **Implement State Manager**: Create `orchestration/state.py`.
    *   Methods: `save(path)`, `load(path)`.
2.  **Implement Loop Logic**: Expand `Orchestrator.run()`.
    *   Add detailed logging ("Starting Cycle X...").
    *   Add error handling (try/except blocks around component calls).
3.  **Refine CLI**: Update `main.py` to handle the `resume` flag.
4.  **Wiring**: Ensure that the output of one component matches the input of the next (e.g., Explorer output is compatible with Oracle input).

## 5. Test Strategy

### 5.1. End-to-End Test (`test_loop.py`)
This is the most critical test.
*   **Setup**: Use **Mocks** for all components to avoid running heavy binaries.
    *   `MockExplorer` returns 5 atoms.
    *   `MockOracle` labels them.
    *   `MockTrainer` produces a dummy file.
    *   `MockValidator` always passes.
*   **Execution**: Run `orchestrator.run()`.
*   **Asserts**:
    *   `current_cycle` reaches `max_cycles`.
    *   `state.json` is created.
    *   `dataset` grows in size each cycle.

### 5.2. Resume Test
*   Run the loop for 1 cycle.
*   Interrupt (simulate crash).
*   Run again with `resume=True`.
*   **Assert**: It starts from Cycle 2, not Cycle 1.
