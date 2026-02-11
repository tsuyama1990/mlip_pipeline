# Cycle 06 Specification: On-the-Fly (OTF) Loop

## 1. Summary

Cycle 06 integrates all previously developed components into a cohesive **On-the-Fly (OTF) Active Learning Loop**. This is the core intelligence of PyAceMaker. The system autonomously manages the "Halt & Diagnose" workflow: detecting when a simulation enters an uncertain region, diagnosing the cause, generating corrective training data, retraining the potential, and resuming the simulation.

This cycle focuses on the **Orchestrator Logic** and the data flow between the Dynamics Engine, Generator, Oracle, and Trainer. It transforms the linear "Generate -> Train" pipeline into a dynamic, self-correcting cycle capable of handling long-timescale simulations without human intervention.

## 2. System Architecture

We update the `core/orchestrator.py` and introduce loop-specific logic.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```
.
├── src/
│   └── mlip_autopipec/
│       ├── core/
│       │   ├── **orchestrator.py**    # Implement `run_otf_loop()`
│       │   ├── **candidate_generator.py** # Local candidate strategy
│       │   └── **resume_logic.py**    # Restart handling
│       └── domain_models/
│           └── **config.py**          # Add OTFConfig (max_retries, strategies)
```

### Key Components
1.  **Orchestrator (`src/mlip_autopipec/core/orchestrator.py`)**: The enhanced controller. It now implements the `run_otf_loop` method which manages the state machine: `EXPLORE -> HALT -> DIAGNOSE -> TRAIN -> RESUME`.
2.  **LocalCandidateGenerator (`src/mlip_autopipec/core/candidate_generator.py`)**: A specialized generator that takes a "Halted Structure" and produces perturbed variations (e.g., Normal Mode sampling or random displacement) to help the model learn the local potential energy surface curvature.
3.  **ResumeManager (`src/mlip_autopipec/core/resume_logic.py`)**: Handles the logistics of restarting a LAMMPS simulation. It ensures that the new potential file is placed correctly and that the restart file (binary or data) is valid.

## 3. Design Architecture

### 3.1. The OTF State Machine
The Orchestrator maintains a state:
*   **EXPLORE**: Running MD/kMC with current potential.
*   **HALT**: Simulation stopped due to high uncertainty ($\gamma > \text{thresh}$).
*   **DIAGNOSE**: Extracting the problematic structure $S_{halt}$.
*   **GENERATE**: Creating local candidates $\{S_i\}$ around $S_{halt}$.
*   **SELECT**: Using D-Optimality to pick the best subset $\{S_{selected}\}$.
*   **LABEL**: Running DFT on $\{S_{selected}\}$ (with periodic embedding).
*   **TRAIN**: Updating the potential with new data.
*   **RESUME**: Restarting simulation with the new potential.

### 3.2. Local Candidate Strategy
When a halt occurs, simply adding the single point $S_{halt}$ is often insufficient. We need to learn the *curvature* (forces) in the vicinity to prevent the simulation from falling back into the same "hole".
*   **Strategy**: Generate $N=10\sim20$ variations.
*   **Methods**:
    *   **Random Displacement**: $\Delta r \sim N(0, 0.05 \AA)$.
    *   **Normal Mode**: Estimate Hessian (using the bad potential or a universal one) and displace along soft modes.
*   **Selection**: Use `ActiveSetSelector` to pick the most informative 5-10 structures from these candidates, ensuring $S_{halt}$ is included.

### 3.3. Resume Logic
*   **LAMMPS**: Use `read_restart` or `read_data` from the halt step.
*   **Potential**: The `pair_coeff` command in the new input script must point to the *newly trained* `.yace` file (e.g., `potential_v2.yace`).

## 4. Implementation Approach

1.  **Orchestrator Update**: Rewrite `run_loop` to support the OTF logic. Use a `while` loop with a state variable.
2.  **Halt Handling**: Implement `diagnose_halt(result)`. It extracts the structure from the dump file at the specific step.
3.  **Local Generation**: Implement `LocalCandidateGenerator`. Start with simple random displacements.
4.  **Integration**: Connect the Generator output to the Oracle (Embedding) -> Trainer (Active Set) -> Dynamics (Resume).
5.  **Config**: Add `OTFConfig` to control parameters like `n_candidates`, `perturbation_scale`, `max_otf_iterations`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **State Machine**: Verify that `run_otf_loop` transitions correctly between states (e.g., `EXPLORE` -> `HALT` -> `DIAGNOSE`).
*   **Candidate Generation**: Verify that `generate_local_candidates` produces distinct structures close to the original.

### 5.2. Integration Testing (Mock Loop)
*   **Goal**: Verify the entire self-healing cycle.
*   **Procedure**:
    1.  Start a Mock Dynamics run configured to halt at step 100.
    2.  The Orchestrator should detect the halt.
    3.  It should generate local candidates (Mock Generator).
    4.  It should "run DFT" (Mock Oracle).
    5.  It should "train" (Mock Trainer) and produce `potential_v2.yace`.
    6.  It should restart Dynamics.
    7.  **Assert**: The loop completes successfully, and the final potential version is incremented.
