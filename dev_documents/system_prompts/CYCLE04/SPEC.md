# Cycle 04 Specification: Active Learning Loop

## 1. Summary
Cycle 04 integrates the components built in previous cycles into a closed-loop **Active Learning** system. The core logic involves the **Orchestrator** managing a state machine that cycles through Exploration (MD), Detection (Halt), Selection (Cluster Extraction), Calculation (DFT), and Refinement (Training). A key technical feature implemented here is **Periodic Embedding**, which allows cutting out a local cluster from a large MD snapshot and preparing it for DFT by wrapping it in a valid periodic supercell.

## 2. System Architecture

### 2.1. File Structure
```text
src/mlip_autopipec/
├── orchestration/
│   ├── loop_manager.py             # [CREATE] Main loop logic
│   └── strategies.py               # [CREATE] Selection strategies
├── dft/
│   └── embedding.py                # [CREATE] Periodic Embedding
└── analysis/                       # [CREATE]
    └── active_set.py               # [CREATE] D-Optimality selection
```

### 2.2. Component Interaction
- **`LoopManager`**: The conductor. It calls `LammpsRunner`, checks for halts. If halted, it calls `Embedding` to extract structures, then `QERunner` for labels, then `PacemakerWrapper` for training.
- **`PeriodicEmbedding`**:
    - Input: Large `Atoms` object (e.g., 1000 atoms), list of high-$\gamma$ atom indices.
    - Output: Small `Atoms` object (e.g., 50 atoms) in a box that preserves the local environment of the target atoms under periodic boundary conditions.
- **`ActiveSetSelector`**: Uses Pacemaker's linear algebra tools to select only the most informative structures from the extracted candidates.

## 3. Design Architecture

### 3.1. Periodic Embedding Logic
- **Problem**: Cutting a sphere out of a crystal destroys periodicity, making DFT calculations hard (need vacuum, surface effects).
- **Solution**:
    1. Identify the "Uncertain Core" (atoms with high $\gamma$).
    2. Define a bounding box (Orthorhombic) that encloses the core + $R_{cut}$ + Buffer.
    3. Extract atoms within this box.
    4. Define the box vectors as the new unit cell.
    5. Result: A small periodic system representing the local defect environment.

### 3.2. The Loop State Machine
The system transitions between:
1.  **EXPLORE**: Run MD with latest potential.
2.  **DETECT**: Wait for `fix halt`.
3.  **SELECT**: Extract candidate structures (embedding) and filter (D-optimality).
4.  **LABEL**: Run DFT on selected candidates.
5.  **TRAIN**: Update potential (fine-tuning).
6.  **DEPLOY**: Update `current.yace` and return to EXPLORE.

## 4. Implementation Approach

1.  **Embedding**: Implement the box extraction logic using `ase.geometry`. Ensure minimum image convention is respected.
2.  **Selector**: Wrap `pace_activeset` command or implement a simplified MaxVol selection if feasible in Python.
3.  **Loop Manager**: Implement the `run_cycle()` method. It should persist state to disk (checkpointing) so it can resume if interrupted.

## 5. Test Strategy

### 5.1. Unit Testing
- **Embedding**: Create a perfect 5x5x5 crystal. Mark the center atom. Request embedding. Verify the output is a smaller valid crystal (e.g., 3x3x3 equivalent) and not a cluster in vacuum.
- **Loop State**: Verify that the manager correctly transitions states (e.g., if DFT fails, go to ERROR state or RETRY).

### 5.2. Integration Testing
- **Mini-Loop**: Run the full cycle on a toy system (e.g., 4 atoms).
    - Step 1: MD runs.
    - Step 2: Manually trigger a "halt" by injecting a high gamma.
    - Step 3: Verify a new dataset file is created.
    - Step 4: Verify `pace_train` is called.
