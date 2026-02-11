# Cycle 06 Specification: On-the-Fly (OTF) Active Learning Loop

## 1. Summary

Cycle 06 is the "Integration Cycle". It takes the components built in Cycles 01-05 and connects them into the autonomous "Active Learning Loop". The core logic resides in the `Orchestrator`, which now manages the state transitions between Exploration (Dynamics), Labeling (Oracle), and Training (Pacemaker).

A key feature added in this cycle is the **Local Candidate Generation** logic upon a "Halt" event. When the Dynamics Engine stops due to high uncertainty, the system doesn't just label that single snapshot. Instead, it generates a swarm of local variations (random displacements, normal mode perturbations) around that high-uncertainty point and uses D-Optimality to select the best set of structures to label. This ensures the potential learns the "local curvature" of the energy landscape, preventing it from falling into the same hole again.

## 2. System Architecture

### File Structure

Files in **bold** are new or modified in this cycle.

```ascii
src/mlip_autopipec/
├── core/
│   ├── **orchestrator.py**      # Full state machine logic implemented
│   └── **candidate_generator.py** # Local swarm generation logic
├── domain_models/
│   ├── **enums.py**             # LoopStatus (EXPLORING, LABELING, TRAINING)
│   └── config.py                # Updated with OTFConfig
└── main.py                      # Updated CLI to run the full loop
```

### Component Interaction (The Loop)
1.  **EXPLORE**: Orchestrator runs `Dynamics.run_exploration()`.
    -   If converged -> Loop Success (Exit).
    -   If Halted -> Go to **REFINE**.
2.  **REFINE (Step 1)**: `CandidateGenerator` creates 20 variations of the Halt structure.
3.  **REFINE (Step 2)**: `Trainer.select_active_set` selects top 5 informative structures.
4.  **LABEL**: `Oracle.compute` calculates DFT labels for the selected 5 structures.
5.  **TRAIN**: `Trainer.train` updates the potential using the new dataset.
6.  **Loop**: Increment iteration count and go back to **EXPLORE**.

## 3. Design Architecture

### 3.1. OTF Configuration (`domain_models/config.py`)

```python
class OTFConfig(BaseModel):
    max_cycles: int = 50
    initial_exploration_steps: int = 1000
    patience: int = 5  # Stop if no improvement
```

### 3.2. Candidate Generator (`core/candidate_generator.py`)

We need a utility to generate "local swarms".

```python
def generate_local_candidates(structure: Structure, n_samples: int) -> List[Structure]:
    # 1. High-temp MD burst (short)
    # 2. Random displacement (rattle)
    # 3. Normal Mode displacement (if Hessian available - future)
    return [rattle(structure) for _ in range(n_samples)]
```

### 3.3. Orchestrator State Machine (`core/orchestrator.py`)

The `run()` method is expanded.

```python
while self.iteration < config.max_cycles:
    # 1. Explore
    result = self.dynamics.run(...)
    if result.status == "CONVERGED":
        break

    # 2. Handle Halt
    halt_struct = result.structure
    candidates = self.candidate_generator.generate(halt_struct)

    # 3. Label
    new_data = self.oracle.compute(candidates)

    # 4. Train
    self.trainer.update_dataset(new_data)
    new_potential = self.trainer.train(...)

    # 5. Update State
    self.state.current_potential = new_potential
    self.iteration += 1
```

## 4. Implementation Approach

1.  **Implement Candidate Generator**: Create `core/candidate_generator.py`. Start with simple random rattling.
2.  **Update Orchestrator**:
    -   Refactor the `run` method to implement the loop described above.
    -   Add state persistence (save `workflow_state.json`) after each step to allow resuming if crashed.
3.  **CLI Update**: Ensure `mlip-runner run` executes this new loop logic.
4.  **Mocking the Loop**:
    -   Create a "Full Mock" scenario where:
        -   Dynamics halts at step 100.
        -   Oracle returns fake data.
        -   Trainer returns a touched potential file.
        -   Loop repeats 3 times and then "converges".

## 5. Test Strategy

### 5.1. Unit Testing
-   **Candidate Generator**: Verify it returns `n` distinct structures based on the input.
-   **State Transitions**: Test the Orchestrator's internal logic (e.g., if Dynamics returns Halt, next state is Labeling).

### 5.2. Integration Testing (The "Mini-Cycle")
-   **Full Mock Loop**:
    -   Configure `config.yaml` with all Mock components.
    -   Run `mlip-runner run config.yaml`.
    -   Verify that:
        -   `iter_001`, `iter_002` directories are created.
        -   `workflow_state.json` updates.
        -   The process finishes gracefully.
-   This is the most critical test for the entire system so far.
