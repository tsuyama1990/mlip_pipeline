# Cycle 06 Specification: Local Learning Loop

## 1. Summary

Cycle 06 implements the **Local Learning Loop** ("Halt & Diagnose"), the core mechanism of Active Learning in PYACEMAKER. This cycle connects the "Halt" signal from the Dynamics Engine (Cycle 05) back to the Oracle (Cycle 03) and Trainer (Cycle 04) in a targeted, efficient manner.

When a simulation halts due to high uncertainty, simply adding that single snapshot to the training set is often insufficient (it learns a "hole" but not the "shape" of the PES around it). This cycle implements a **Local Candidate Generation** strategy:
1.  **Extract**: Isolate the atomic environment responsible for the high uncertainty.
2.  **Perturb**: Generate a cloud of physically meaningful variations (local candidates) around this environment using Normal Mode Analysis or short MD bursts.
3.  **Select**: Use **Local D-Optimality** to pick a small, optimal subset of these candidates that maximizes information gain.
4.  **Embed**: Prepare these for DFT (Periodic Embedding from Cycle 03).
5.  **Update**: Trigger a rapid "Fine-Tuning" training session.
6.  **Resume**: Restart the simulation with the improved potential.

## 2. System Architecture

Files in **bold** are to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── core/
│   └── **active_learner.py** # Orchestrates the Halt-Diagnose-Train loop
├── domain_models/
│   └── **config.py**         # Update ActiveLearningConfig
├── generator/
│   └── **candidate_generator.py** # Local perturbations (Normal Mode / Random)
├── trainer/
│   └── **active_selector.py** # Local D-Optimality logic
└── tests/
    └── unit/
        └── **test_active_learner.py**
```

## 3. Design Architecture

### 3.1 Active Learner (`core/active_learner.py`)
This class manages the specific workflow triggered by a `HaltEvent`.
*   **Input**: `HaltEvent` (containing snapshot and high-gamma atom indices).
*   **Workflow**:
    *   Calls `CandidateGenerator.generate_local(snapshot, indices)`.
    *   Calls `ActiveSelector.select_batch(candidates, method='maxvol')`.
    *   Calls `Oracle.compute(selected_candidates)`.
    *   Calls `Trainer.fine_tune(new_data)`.
    *   Returns updated `Potential`.

### 3.2 Candidate Generator (`generator/candidate_generator.py`)
Generates variations of a structure.
*   **Method A (Random Displacement)**: Displaces atoms within a radius $R$ by a random vector $\delta$.
*   **Method B (Normal Mode - *Advanced*)**: Calculates Hessian (using the current potential) and displaces along soft modes.
*   **Method C (MD Burst)**: Runs very short, high-T MD trajectories starting from the halt structure.

### 3.3 Local Active Selector (`trainer/active_selector.py`)
Selects the best subset from candidates.
*   **Logic**: Uses the `pace_activeset` command on the small batch of candidates to find the most distinct ones relative to the *current* active set.

## 4. Implementation Approach

1.  **Enhance Domain Models**: Add `ActiveLearningConfig` (e.g., perturbation magnitude, number of candidates).
2.  **Implement CandidateGenerator**: Write logic to take an `Atoms` object and return a list of perturbed `Atoms`.
3.  **Implement ActiveSelector**: Wrapper around `pace_activeset` for small batches.
4.  **Implement ActiveLearner**: The main logic that ties it all together. This will be called by the `Orchestrator` when `Dynamics.run()` returns a Halt.
5.  **Integration**: Wire this into the main Orchestrator loop.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Candidate Generation**: Verify that generated structures are distinct from the original but retain the general motif (not exploded).
*   **Selector Logic**: Feed a set of 10 very similar structures and 1 distinct one. Verify the selector picks the distinct one.

### 5.2 Integration Testing
*   **Simulated Halt Loop**:
    1.  Create a "Fake Halt" event (manually high-gamma structure).
    2.  Pass to `ActiveLearner`.
    3.  Verify it generates candidates -> mock DFT -> mock Train -> new potential.
    4.  Verify the new potential has lower uncertainty on the original structure (using Mock Trainer behavior).
