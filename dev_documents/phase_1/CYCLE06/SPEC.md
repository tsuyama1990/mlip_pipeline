# Cycle 06: The Active Learning Loop

## 1. Summary

We have built the engine (MD), the steering wheel (Policy/Config), the teacher (DFT), the learner (Pacemaker), and the inspector (Validator). Cycle 06 is about building the **Driver**.

This cycle implements the autonomous active learning loop. It transitions the system from a collection of isolated scripts to a state-aware robot that can run for days or weeks without human intervention. The core logic revolves around the **Orchestrator**, which manages the `WorkflowState`.

The loop follows this sequence:
1.  **Exploration**: Run MD with the current potential. Monitor uncertainty ($\gamma$).
2.  **Detection**: If MD halts due to high $\gamma$, identifying the "confusing" atomic configuration.
3.  **Selection**: Extract the local cluster, embed it in a periodic box (Cycle 03), and add it to the candidate list.
4.  **Calculation**: Send candidates to the Oracle for labelling.
5.  **Refinement**: Update the dataset and re-train the potential (Fine-tuning).
6.  **Validation**: Check if the new potential is better. If so, promote it.
7.  **Loop**: Increment generation count and repeat.

A key feature is **Resumability**. If the cluster crashes (power outage, walltime limit), the system must be able to load the `state.json` and resume exactly where it left off, rather than restarting from Generation 0.

## 2. System Architecture

We expand the `orchestration` package.

### File Structure
Files to be created/modified are in **bold**.

```ascii
mlip_autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   ├── **workflow.py**         # WorkflowState schema
│       │   └── **structure.py**        # Update with CandidateStatus
│       ├── orchestration/
│       │   ├── **manager.py**          # The main loop logic
│       │   ├── **state.py**            # Load/Save state
│       │   └── **candidate_processing.py** # Selection logic
│       └── physics/
│           └── dynamics/
│               └── **log_parser.py**   # Detect 'fix halt'
└── tests/
    └── orchestration/
        └── **test_manager.py**
```

### Component Interaction

1.  **CLI** calls `WorkflowManager.run_loop()`.
2.  **`WorkflowManager`** loads `state.json`.
3.  **State Machine**:
    -   `if state.phase == EXPLORATION`:
        -   Call `LammpsRunner`.
        -   Parse logs with `LogParser`.
        -   If halted, extract structure -> `state.candidates`.
        -   `state.phase = SELECTION`.
    -   `if state.phase == SELECTION`:
        -   Filter candidates (remove duplicates).
        -   `state.phase = CALCULATION`.
    -   `if state.phase == CALCULATION`:
        -   Call `QERunner` for pending candidates.
        -   Update `state.dataset_path`.
        -   `state.phase = TRAINING`.
    -   `if state.phase == TRAINING`:
        -   Call `PacemakerRunner`.
        -   Update `state.latest_potential`.
        -   `state.phase = VALIDATION`.
    -   `if state.phase == VALIDATION`:
        -   Call `ValidationRunner`.
        -   If pass, `state.generation += 1`.
        -   `state.phase = EXPLORATION`.

## 3. Design Architecture

### 3.1. Workflow State (`domain_models/workflow.py`)

-   **Enum `WorkflowPhase`**:
    -   `EXPLORATION`, `SELECTION`, `CALCULATION`, `TRAINING`, `VALIDATION`.

-   **Class `WorkflowState`**:
    -   `project_name`: `str`.
    -   `generation`: `int`.
    -   `current_phase`: `WorkflowPhase`.
    -   `latest_potential_path`: `Optional[Path]`.
    -   `dataset_path`: `Path`.
    -   `candidates`: `List[CandidateStructure]`.

### 3.2. Candidate Processing
We need to handle the transition from a "Exploded MD Frame" to a "Trainable Cluster".
-   **Class `CandidateManager`**:
    -   `extract_cluster(supercell, center_atom_id, radius)`: Cuts a sphere.
    -   `embed_cluster(cluster)`: Wraps it in a box.

### 3.3. Log Parsing (`physics/dynamics/log_parser.py`)
-   **Class `LammpsLogParser`**:
    -   Must detect the specific line: `ERROR: Fix halt ...`.
    -   Extract the timestep.

## 4. Implementation Approach

### Step 1: State Management
-   Implement `domain_models/workflow.py`.
-   Implement `orchestration/state.py` to save `WorkflowState` to `state.json` (atomic write to avoid corruption).

### Step 2: Log Parser
-   Implement `physics/dynamics/log_parser.py`.
-   Use regex to parse LAMMPS logs.

### Step 3: Candidate Processor
-   Implement `candidate_processing.py`.
-   Use `ase.neighborlist` to find neighbors of the high-uncertainty atom.

### Step 4: The Manager (The Brain)
-   Implement `WorkflowManager`.
-   Write the big `while` loop.
-   Add extensive logging. "Transitioning from EXPLORATION to SELECTION".

## 5. Test Strategy

### 5.1. Unit Testing
-   **State Logic**:
    -   Load a state at `CALCULATION`.
    -   Call `manager.step()`.
    -   Verify it calls the Oracle and transitions to `TRAINING`.

-   **Log Parser**:
    -   Feed a log string containing `Fix halt condition met`.
    -   Assert `parser.detected_halt` is True.

### 5.2. Integration Testing (Simulated Loop)
-   **Mock Everything**:
    -   Mock `LammpsRunner` to always "halt" after 10 steps.
    -   Mock `QERunner` to always return energy=0.
    -   Mock `PacemakerRunner` to always produce a new `.yace`.
-   **Run**:
    -   Initialize loop at Gen 0.
    -   Run for 2 full cycles.
    -   Assert `generation` reaches 2.
    -   Assert `state.json` is updated.

### 5.3. Pre-commit
-   Check cyclomatic complexity of `run_loop`. It might get complex, so decompose it into `_run_exploration()`, `_run_training()`, etc.
