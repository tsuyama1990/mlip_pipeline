# Cycle 04 Specification: Trainer & Pacemaker Integration

## 1. Summary
This cycle integrates **Pacemaker** (Atomic Cluster Expansion) to train Machine Learning Interatomic Potentials. It implements the `Trainer` component, which manages the training loop, including **Delta Learning** (learning the difference between DFT and a physical baseline like Lennard-Jones/ZBL) and **Active Set Optimization** (selecting the most informative structures).

## 2. System Architecture

### 2.1. File Structure
The following file structure must be created/modified. **Bold** files are to be implemented in this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── trainer/
│   │   ├── **__init__.py**
│   │   ├── **base.py**     # Base Trainer
│   │   ├── **pacemaker.py** # Pacemaker Wrapper
│   │   └── **delta_learning.py** # Baseline Potential Logic
tests/
    └── **test_trainer.py**
```

### 2.2. Class Blueprints

#### `src/mlip_autopipec/components/trainer/pacemaker.py`
```python
from pathlib import Path
from mlip_autopipec.components.trainer.base import BaseTrainer

class PacemakerTrainer(BaseTrainer):
    def train(self, dataset: Path, workdir: Path) -> Path:
        """
        Run pace_train via CLI or Python API.
        Returns path to generated potential.yace.
        """
        pass

    def select_active_set(self, structures: Iterator[Structure]) -> List[Structure]:
        """
        Run pace_activeset (MaxVol) to filter structures.
        """
        pass
```

## 3. Design Architecture

### 3.1. Trainer Configuration
*   **`TrainerConfig`**:
    *   `dataset_path`: Path to the training set.
    *   `max_num_epochs`: int (e.g., 50 for fine-tuning, 1000 for full training).
    *   `batch_size`: int.
    *   `energy_loss`: float (weight).
    *   `force_loss`: float (weight).
    *   `stress_loss`: float (weight).

### 3.2. Delta Learning Setup
*   **Goal**: Ensure physical robustness at short range.
*   **Implementation**:
    *   Define a reference potential `V_ref` (e.g., ZBL or LJ).
    *   Pacemaker learns `V_ACE = V_DFT - V_ref`.
    *   Configuration requires creating a `input.yaml` for Pacemaker that specifies the reference potential.

## 4. Implementation Approach

1.  **Pacemaker Wrapper**: Implement `PacemakerTrainer.train()` using `subprocess.run(["pace_train", ...])`.
2.  **Config Generation**: Implement a helper to generate `input.yaml` for Pacemaker based on `TrainerConfig`.
3.  **Active Set**: Implement `select_active_set` using `pace_activeset`. This will be used by the Orchestrator to filter data before training.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_trainer.py` (Mocked)**:
    *   Mock `subprocess.run`.
    *   Verify that `pace_train` command is constructed correctly with all flags.
    *   Verify `input.yaml` content (especially the reference potential section).

### 5.2. Integration Testing
*   **Real Training (Requires Pacemaker)**:
    *   Create a small dataset (10 structures).
    *   Run `PacemakerTrainer.train()`.
    *   Verify `potential.yace` is created.
    *   Load `potential.yace` with ASE/LAMMPS (if available) to ensure it's valid.
