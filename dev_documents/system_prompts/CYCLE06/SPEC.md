# Cycle 06 Specification: Active Learning Orchestrator

## 1. Summary
Cycle 06 implements the "Brain" of the system: the **Orchestrator**. It ties together the Generator, Oracle, Trainer, and Inference Engine into a continuous, self-healing loop. This cycle defines the state machine that manages the transition from Exploration to Refinement and ensures the workflow can be paused and resumed without data loss.

## 2. System Architecture

```ascii
mlip_autopipec/
├── config/
│   └── schemas/
│       └── workflow.py         # **Workflow Config**
├── data_models/
│   └── state.py                # **WorkflowState Model**
└── orchestration/
    ├── loop.py                 # **WorkflowManager Class**
    └── strategies.py           # **Active Learning Strategies**
```

## 3. Design Architecture

### 3.1. Workflow State (`data_models/state.py`)
A Pydantic model serialized to JSON/YAML to track progress.
- `cycle_index`: int
- `current_phase`: Enum (`Exploration`, `Selection`, `Calculation`, `Training`)
- `latest_potential_path`: Path
- `active_tasks`: List[IDs]

### 3.2. Workflow Manager (`orchestration/loop.py`)
The main controller class.
- `run()`: The entry point.
- **Logic**:
  ```python
  while not converged:
      if phase == Exploration:
          result = run_md()
          if result.halted: phase = Selection
      elif phase == Selection:
          candidates = select_candidates(result.dump)
          phase = Calculation
      elif phase == Calculation:
          new_data = run_dft(candidates)
          phase = Training
      elif phase == Training:
          potential = run_training(new_data)
          phase = Exploration
  ```

### 3.3. Selection Strategy
Logic to pick which atoms from a halted MD dump need DFT.
- **Strategy**: Filter by $\gamma$ > threshold. Use `pace_activeset` to downsample.
- **Embedding**: Call the Cycle 03 Embedding utility here to convert clusters to supercells.

## 4. Implementation Approach

1.  **State Model**: Define `WorkflowState`.
2.  **Manager**: Implement `WorkflowManager` with injected dependencies (Runners from previous cycles).
3.  **Persistence**: Ensure state is saved after every phase transition.
4.  **Integration**: Wire up `LammpsRunner` -> `Selection` -> `QERunner` -> `PacemakerWrapper`.

## 5. Test Strategy

### 5.1. Unit Testing
- **State Transitions**: Verify logic flow. e.g., `Exploration` -> `Selection` on halt.
- **Persistence**: Save/Load state and verify integrity.

### 5.2. Integration Testing (Mocked)
- **Full Loop**:
    1.  Mock MD to halt immediately.
    2.  Mock DFT to return random energies.
    3.  Mock Training to produce a new potential file.
    4.  Run `WorkflowManager`.
    5.  Verify it completes one full cycle and updates the `latest_potential_path`.
