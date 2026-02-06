# Cycle 07 Specification: Advanced Exploration & Robustness

## 1. Summary
This cycle upgrades the "Explorer" and "Orchestrator" to handle the complexities of real-world materials discovery. We introduce two critical features: **Uncertainty-Driven Exploration** (using the `fix halt` command in LAMMPS to stop simulations that enter unknown territory) and **Adaptive Structure Generation** (a policy engine that adjusts simulation parameters like temperature and pressure based on the current model's confidence). We also implement robust error recovery, ensuring the pipeline doesn't crash when DFT calculations fail or when the potential becomes unstable.

## 2. System Architecture

### 2.1. File Structure

```
mlip-pipeline/
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   └── **policy.py**           # [NEW] ExplorationPolicy model
│       ├── orchestration/
│       │   └── **policies.py**         # [NEW] Adaptive policy logic
│       ├── infrastructure/
│       │   ├── lammps/
│       │   │   └── **watchdog.py**     # [NEW] Halt & Diagnose logic
│       │   └── **recovery.py**         # [NEW] General error handling
└── tests/
    └── unit/
        └── **test_policy.py**          # [NEW] Tests for adaptive policies
```

## 3. Design Architecture

### 3.1. `ExplorationPolicy` (Pydantic)
Dynamically generated configuration for the Explorer.
*   `temperature_schedule`: List[float].
*   `pressure_schedule`: List[float].
*   `max_steps`: int.
*   `uncertainty_threshold`: float (The $\gamma$ value that triggers a halt).

### 3.2. Uncertainty Watchdog (LAMMPS Integration)
The ACE potential provides an extrapolation grade $\gamma$. We must configure LAMMPS to monitor this.
*   **Command**: `fix stop all halt 10 v_gamma > 5.0 error hard`
*   **Behavior**: If $\gamma > 5.0$, LAMMPS exits with a specific error code.
*   **Diagnosis**: The Python wrapper must catch this error, read the dump file, find the frame where the halt occurred, and return *that specific structure* as a high-priority candidate for labeling.

### 3.3. Adaptive Policy Engine
Instead of random sampling, the Orchestrator uses a `PolicyEngine`.
*   **Input**: Current Cycle, Validation RMSE, Previous Halt Rate.
*   **Logic**:
    *   If Halt Rate is high -> Reduce Temperature, Increase Sampling Density (Cautious Mode).
    *   If Halt Rate is low -> Increase Temperature, Explore High Pressure (Aggressive Mode).

## 4. Implementation Approach

1.  **Enhance LAMMPS Templates**: Update `infrastructure/lammps/templates.py` to include the `compute pace`, `variable gamma`, and `fix halt` commands if `uncertainty_threshold` is set.
2.  **Implement Watchdog Handler**: In `LammpsDynamics.run()`, wrap the execution in a try/except block.
    *   If exit code indicates Halt:
        *   Parse the log to find the timestep.
        *   Extract the structure at that timestep.
        *   Return it with a flag `reason="high_uncertainty"`.
3.  **Implement Policy Engine**: Create `orchestration/policies.py`.
    *   Define a class `AdaptivePolicy`.
    *   Method `get_next_config(stats) -> ExplorerConfig`.
4.  **Integration**: Update `Orchestrator` to use the Policy Engine to set the `ExplorerConfig` for each cycle.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Policy Logic**:
    *   Feed stats with "high halt rate".
    *   **Assert**: `get_next_config()` returns lower temperature.
    *   Feed stats with "low RMSE".
    *   **Assert**: `get_next_config()` returns aggressive parameters.

### 5.2. Integration Testing (Simulated Halt)
*   **Mocking LAMMPS Halt**:
    *   Create a Mock Explorer that simulates a "Halt" event (returns a specific exit code and a dummy structure).
    *   **Assert**: The Orchestrator correctly identifies this as a "Discovery" and prioritizes the structure for labeling.
    *   **Assert**: The Policy Engine adapts the next cycle's parameters (e.g., becomes more cautious).
