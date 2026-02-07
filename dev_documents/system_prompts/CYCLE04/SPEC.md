# Cycle 04 Specification: The Trainer (Pacemaker Integration)

## 1. Summary
This cycle integrates the machine learning engine, `Pacemaker` (ACE formalism), into the pipeline. The goal is to train a high-accuracy interatomic potential from the accumulated DFT data. Key features include "Delta Learning" (learning the difference between DFT and a physics baseline like LJ/ZBL) to ensure robustness, and "Active Set" optimization (D-optimality) to select the most informative structures for training, keeping computational costs manageable.

## 2. System Architecture

### 2.1. File Structure
The following files must be created or modified. **Bold** files are the focus of this cycle.

src/mlip_autopipec/
├── domain_models/
│   ├── **potential.py**            # Enhanced Potential Model (path to .yace)
│   ├── **config.py**               # TrainerConfig (PacemakerConfig)
├── interfaces/
│   ├── **trainer.py**              # Enhanced BaseTrainer
├── infrastructure/
│   ├── **trainer/**
│   │   ├── **__init__.py**
│   │   ├── **pacemaker_wrapper.py** # Wrapper for pace_train/pace_activeset
│   │   ├── **delta_learning.py**   # LJ/ZBL Baseline Generator
│   │   └── **active_set.py**       # Active Set Selection Logic
└── orchestrator/
    └── **simple_orchestrator.py**  # Update logic to call Trainer

### 2.2. Class Diagram
*   `PacemakerTrainer` implements `BaseTrainer`.
*   `ActiveSetSelector` handles `pace_activeset` execution.
*   `DeltaLearningManager` handles baseline potential generation.

## 3. Design Architecture

### 3.1. Trainer Logic (`infrastructure/trainer/pacemaker_wrapper.py`)
*   **Input**: `Dataset` (accumulated structures), `BasePotential` (optional, for fine-tuning).
*   **Process**:
    1.  **Baseline**: Generate or load LJ/ZBL parameters for the elements involved.
    2.  **Active Set**: Select optimal structures using MaxVol algorithm (`pace_activeset`).
    3.  **Config**: Generate `input.yaml` for `pace_train`.
        *   Set `cutoff`, `order`, `loss weights` (Energy/Force/Stress).
        *   Enable `delta_learning` if baseline is provided.
    4.  **Train**: Execute `pace_train` via `subprocess`.
    5.  **Output**: `Potential` object pointing to the new `.yace` file.

### 3.2. Delta Learning (`infrastructure/trainer/delta_learning.py`)
*   **Goal**: Ensure core repulsion and long-range physics.
*   **Logic**:
    *   For each element pair (e.g., Fe-Fe, Fe-Pt), generate ZBL parameters (Ziegler-Biersack-Littmark).
    *   Alternatively, fit Lennard-Jones parameters to initial dimers.
    *   Write these to a `potential_baseline.yace` or YAML format compatible with Pacemaker.

### 3.3. Active Set (`infrastructure/trainer/active_set.py`)
*   **Goal**: Reduce training set size from N (10,000) to n (500) without losing information.
*   **Logic**:
    *   Compute descriptors for all structures.
    *   Compute the Information Matrix (Gram Matrix).
    *   Select `n` structures that maximize the determinant (D-optimality).
    *   Return indices of selected structures.

## 4. Implementation Approach

1.  **Dependencies**: Ensure `pacemaker` is installed (or mocked).
2.  **Implement DeltaLearning**: Create functions to generate ZBL/LJ potentials.
3.  **Implement ActiveSet**: Wrap `pace_activeset` command.
4.  **Implement Trainer**: Wrap `pace_train` command.
5.  **Update Config**: Add `TrainerConfig` for hyperparameters (cutoff, max_epochs).

## 5. Test Strategy

### 5.1. Unit Testing
*   **Config Generation**: Verify `PacemakerTrainer` creates valid `input.yaml`.
*   **Command Construction**: Verify correct flags are passed to `pace_train`.
*   **Baseline Generation**: Verify ZBL parameters are correct for given elements.

### 5.2. Integration Testing
*   **Mock Training**:
    *   Create a small dataset (10 atoms).
    *   Run `PacemakerTrainer.train()`.
    *   If `pacemaker` binary is missing, use a mock script that writes a dummy `.yace`.
    *   Verify output file exists and is a valid YAML/YACE file.
