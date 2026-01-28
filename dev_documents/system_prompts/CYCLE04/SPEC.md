# Cycle 04 Specification: Active Learning Loop

## 1. Summary

Cycle 04 is the integration phase where the "Brain" (Orchestrator) connects all previous modules (Oracle, Generator, Trainer, Dynamics) into a self-driving loop. The focus is on automating the "Active Learning Cycle": Exploration -> Detection -> Selection -> Calculation -> Refinement. This cycle introduces the logic to manage iterations, resume from checkpoints, and handle the data flow between LAMMPS dumps, ASE structures, and Pacemaker datasets. Specifically, it implements the "Local D-Optimality" selection to choose the most informative structures from halted simulations.

## 2. System Architecture

Files to be added/modified (in bold):

```ascii
mlip_autopipec/
├── **orchestration/**
│   ├── **workflow.py**        # Orchestrator class (WorkflowManager)
│   └── **candidate.py**       # Candidate selection logic (CandidateProcessor)
├── **domain_models/**
│   ├── **state.py**           # WorkflowState class
│   └── **candidate.py**       # Candidate config/models
├── **dft/**
│   └── **embedding.py**       # PeriodicEmbedding logic
├── **app.py**                 # Full "run-loop" command
└── ...
```

## 3. Design Architecture

### 3.1 Orchestrator Logic

**`WorkflowManager` (in `orchestration/workflow.py`)**
-   **Responsibilities**:
    -   Initialize the loop from `WorkflowConfig`.
    -   Maintain `WorkflowState` (current iteration, current potential path).
    -   Execute the state machine:
        1.  `run_exploration()`: Calls `LammpsRunner` (via `ExplorationPhase`).
        2.  `process_candidates()`: If halted, calls `CandidateProcessor` (via `SelectionPhase`).
        3.  `run_oracle()`: Calls `QERunner` (via `DFTPhase`).
        4.  `run_training()`: Calls `PacemakerWrapper` (via `TrainingPhase`).
-   **Checkpointing**: Saves `state.json` after every step to allow crash recovery.

### 3.2 Candidate Processor

**`CandidateProcessor` (in `orchestration/candidate.py`)**
-   **Responsibilities**:
    -   Extract the halted structure from LAMMPS dump.
    -   Generate "Local Candidates" (perturbations around the halted structure).
    -   Select the best candidates using `pace_activeset` (D-Optimality).
    -   Apply "Periodic Embedding": Cut a cluster and wrap it in a box for DFT.

### 3.3 Workflow State

**`WorkflowState` (in `domain_models/state.py`)**
-   **Fields**:
    -   `cycle_index`: int
    -   `current_phase`: WorkflowPhase (EXPLORATION, SELECTION, CALCULATION, TRAINING)
    -   `latest_potential_path`: Path
    -   `dataset_path`: Path
    -   `halted_structures`: list[Path] (To track dumps found during exploration)

## 4. Implementation Approach

1.  **State Management**:
    -   Update `WorkflowState` using Pydantic. Ensure it can be serialized to JSON.
    -   Implement logic to load state on startup (Resume capability) in `WorkflowManager`.

2.  **Selection Logic**:
    -   Wrap `pace_activeset` command in `PacemakerWrapper`. It takes a pool of structures and an existing dataset/potential, and returns indices of structures that maximize the information gain.
    -   Implement `CandidateProcessor` to coordinate extraction, perturbation, selection, and embedding.

3.  **The Loop**:
    -   Implement the `while` loop in `WorkflowManager.run()`.
    -   Use `shutil` or `pathlib` to organize directories (`work_dir/iter_001/`, etc.).

4.  **Integration**:
    -   Wire up `LammpsRunner` (in `inference/runner.py`) -> `CandidateProcessor` -> `QERunner` -> `DatabaseManager` -> `PacemakerWrapper`.

## 5. Test Strategy

### 5.1 Unit Testing
-   **State Serialization**: Save and load `WorkflowState`. Verify paths are correct.
-   **Candidate Processing**: Mock the `pace_activeset` call. Verify that `CandidateProcessor` returns a subset of input structures.
-   **Embedding**: Verify `ClusterEmbedder` correctly cuts clusters and applies periodicity.

### 5.2 Integration Testing
-   **Mini-Loop**:
    -   Mock the Oracle (return random energies) and Dynamics (return immediate halt).
    -   Run the Orchestrator for 2 iterations.
    -   Verify directory structure: `iter_000`, `iter_001` exist.
    -   Verify `potential.yace` is updated (timestamp changes).
-   **Resume Test**:
    -   Run loop, interrupt it (Ctrl-C or kill).
    -   Restart.
    -   Verify it detects the existing state and continues, rather than starting over.
