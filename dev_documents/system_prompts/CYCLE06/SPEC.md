# Cycle 06 Specification: Orchestrator & Active Learning Loop

## 1. Summary
**Goal**: Implement the `Orchestrator`, the central nervous system of PYACEMAKER. This cycle integrates all previously built components (Generator, Oracle, Trainer, Dynamics) into a coherent, automated **Active Learning Loop**.

**Key Features**:
*   **Orchestrator Logic**: Implements the main `while` loop: Explore -> Detect -> Select -> Label -> Train -> Deploy.
*   **Candidate Selection**: Filters halted structures and selects the most informative ones (D-Optimality).
*   **State Management**: Resumes interrupted workflows seamlessly.
*   **File Organization**: Keeps `active_learning/iter_XXX` directories clean and organized.

## 2. System Architecture

Files to be implemented/modified are **bold**.

```ascii
src/mlip_autopipec/
├── orchestrator/
│   ├── **__init__.py**
│   ├── **workflow.py**         # Main Loop
│   └── **candidate_selector.py** # Filtering Logic
├── **main.py**                 # Update CLI to run workflow
└── tests/
    └── **test_orchestrator/**
        └── **test_workflow.py**
```

## 3. Design Architecture

### 3.1. Orchestrator Component (`src/mlip_autopipec/orchestrator/`)

#### `workflow.py`
*   **`Orchestrator`**:
    *   **Members**: `Generator`, `Oracle`, `Trainer`, `Dynamics`, `State`.
    *   **`run()`**:
        ```python
        while self.state.iteration < self.config.max_iterations:
            # 1. Check for Halt / New Structures
            result = self.dynamics.run(self.state.current_potential)
            if result.halted:
                # 2. Select Candidates (Active Learning)
                candidates = self.selector.select(result.final_structure)
                # 3. Label (DFT)
                labeled_data = self.oracle.compute(candidates)
                # 4. Update Dataset
                self.trainer.update_dataset(labeled_data)
                # 5. Train
                new_pot = self.trainer.train(self.state.dataset_path)
                # 6. Deploy
                self.state.current_potential = new_pot
                self.state.iteration += 1
            else:
                log("Simulation converged without high uncertainty.")
                break
        ```

#### `candidate_selector.py`
*   **`CandidateSelector`**:
    *   **Input**: `HaltedStructure` (high $\gamma$ snapshot).
    *   **Process**:
        1.  Extract local clusters around high-uncertainty atoms.
        2.  Generate perturbations (Normal Mode / Random Displacement).
        3.  Call `Trainer.select_active_set()` (D-Optimality).
    *   **Output**: List of `Structure` ready for Oracle.

### 3.2. Integration Updates

*   **`main.py`**: Update the `run` command to instantiate `Orchestrator` and call `run()`.

## 4. Implementation Approach

1.  **Define Orchestrator Class**: Initialize all sub-components based on `config.yaml`.
2.  **Implement Candidate Selector**: Connect `Dynamics.halt_info` -> `Generator.perturb` -> `Trainer.active_set`.
3.  **Implement Main Loop**: Write the logic flow, handling exceptions and state saving.
4.  **Mock Integration**: Test the entire loop using Mock components for everything (MockDFT returns random energy, MockTrainer returns dummy potential).

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_workflow.py`**:
    *   Mock all dependencies.
    *   Assert that `oracle.compute` is called when `dynamics` returns `halted=True`.
    *   Assert that state is saved after each iteration.

### 5.2. Integration Testing
*   **End-to-End Mock**:
    *   Run `mlip-runner run config.yaml` with `IS_CI_MODE=True`.
    *   Verify directories `active_learning/iter_000`, `iter_001` are created.
    *   Verify `workflow_state.json` updates correctly.
