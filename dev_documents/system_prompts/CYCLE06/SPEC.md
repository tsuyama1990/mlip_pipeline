# Cycle 06 Specification: OTF Loop Integration

## 1. Summary
This cycle is the "Brain Surgery" phase. We connect the isolated components—Structure Generator, Oracle, Trainer, and Dynamics Engine—into a coherent, self-correcting feedback loop known as the **On-the-Fly (OTF) Loop**. The logic is complex: instead of a linear pipeline, the system must handle interrupts. When the Dynamics Engine halts due to high uncertainty, the Orchestrator must capture the state, generate local candidates around the failure point (using small perturbations or normal mode sampling), send these to the Oracle for labeling, retrain the potential, and then resume the simulation. This cycle enables the "Autonomous" aspect of the system.

## 2. System Architecture

### 2.1. File Structure
files to be created/modified in this cycle are bolded.

```text
src/mlip_autopipec/
├── core/
│   ├── orchestrator.py             # [MODIFY] Implement full state machine
│   └── candidate_generator.py      # [CREATE] Local candidate logic
├── components/
│   └── generator/
│       └── utils.py                # [MODIFY] Add normal mode logic
└── domain_models/
    └── enums.py                    # [MODIFY] Add LoopStatus
```

### 2.2. Component Interaction
1.  **`Orchestrator.run()`**:
    *   Enters `DYNAMICS` phase.
    *   Calls `DynamicsEngine.run_exploration()`.
2.  **`DynamicsEngine`**:
    *   Returns `HALTED` and `halt_structure`.
3.  **`Orchestrator`** (Interrupt Handler):
    *   Calls `CandidateGenerator.generate_local_candidates(halt_structure)`.
    *   Calls `Trainer.select_active_set()` (Local selection).
    *   Calls `Oracle.compute()`.
    *   Calls `Trainer.train()` (Fine-tuning).
    *   Resumes `DynamicsEngine` from the halt point (or restarts).

## 3. Design Architecture

### 3.1. Core Logic

#### `orchestrator.py` (Refined State Machine)
*   **State**: `LoopStatus` (`CONVERGED`, `LEARNING`, `MAX_ITER_REACHED`).
*   **Logic**:
    ```python
    while iteration < max_iterations:
        result = dynamics.run()
        if result.status == CONVERGED:
            break
        elif result.status == HALTED:
            # The "Learning Event"
            candidates = candidate_gen.enhance(result.structure)
            labeled = oracle.compute(candidates)
            trainer.update(labeled)
            trainer.train(fine_tune=True)
    ```

#### `candidate_generator.py`
*   **Responsibility**: Don't just label the single halted structure (too little info). Label the *region*.
*   **Strategy**:
    *   **Input**: `HaltStructure` (Atoms).
    *   **Action**: Generate 10-20 perturbations:
        *   Random displacements ($0.05 \AA$).
        *   Scale volume ($\pm 2\%$).
        *   (Advanced) Calculate Hessian with current potential and move along soft modes.

## 4. Implementation Approach

### Step 1: Candidate Generator
*   Implement `src/mlip_autopipec/core/candidate_generator.py`.
*   Implement `generate_local_candidates(atoms, n_samples)`.

### Step 2: Orchestrator Loop Logic
*   Refactor `Orchestrator.run()` to handle the conditional loop.
*   Implement the "Halt -> Retrain -> Resume" sequence.
*   Ensure state is saved after every micro-step (to prevent re-calculating DFT if process dies).

### Step 3: Fine-Tuning Support
*   Update `Trainer.train()` to accept `initial_potential` argument.
*   If `initial_potential` is provided, Pacemaker should load it and run fewer epochs (fine-tuning).

### Step 4: Resume Logic
*   Ensure `DynamicsEngine` can accept a `restart_file` to continue MD from where it left off (optional for Cycle 06, restart from zero is acceptable for simplicity).

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_candidate_gen.py`**:
    *   Input: 1 Atom.
    *   Action: Generate 5 candidates.
    *   Assert: 5 Atoms objects returned, positions slightly different.

### 5.2. Integration Testing
*   **`test_otf_loop_mock.py`**:
    *   **Setup**: Mock Dynamics (halts once), Mock Oracle (returns forces), Mock Trainer (updates timestamp).
    *   **Action**: Run Orchestrator.
    *   **Expectation**:
        1.  Start Dynamics.
        2.  Detect Halt.
        3.  Call Oracle.
        4.  Call Trainer.
        5.  Restart Dynamics.
        6.  Dynamics finishes (mocked to succeed on 2nd try).
        7.  Loop completes.
