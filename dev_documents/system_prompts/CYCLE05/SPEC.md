# Cycle 05 Specification: Dynamics Engine & On-the-Fly Learning

## 1. Summary
This cycle implements the **Dynamics Engine**, which drives the active learning process. Using **LAMMPS** (Large-scale Atomic/Molecular Massively Parallel Simulator), it performs Molecular Dynamics (MD) simulations to explore the configuration space. The critical feature is the **On-the-Fly (OTF) Loop**, where the simulation is automatically halted when the potential's uncertainty (extrapolation grade $\gamma$) exceeds a safe threshold.

## 2. System Architecture

### 2.1. File Structure
The following file structure must be created/modified. **Bold** files are to be implemented in this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── dynamics/
│   │   ├── **__init__.py**
│   │   ├── **base.py**     # Base Dynamics
│   │   └── **lammps_engine.py** # LAMMPS Wrapper
│   └── **otf.py**          # Active Learning Loop Logic
tests/
    └── **test_dynamics.py**
```

### 2.2. Class Blueprints

#### `src/mlip_autopipec/components/dynamics/lammps_engine.py`
```python
from mlip_autopipec.components.dynamics.base import BaseDynamics

class LammpsDynamics(BaseDynamics):
    def run_exploration(self, potential_path: Path, workdir: Path) -> ExplorationResult:
        """
        Run MD using LAMMPS Python interface or subprocess.
        Configures 'pair_style hybrid/overlay pace zbl'.
        Configures 'fix halt' on max_gamma > threshold.
        """
        pass
```

## 3. Design Architecture

### 3.1. Dynamics Configuration
*   **`DynamicsConfig`**:
    *   `temperature`: float.
    *   `steps`: int.
    *   `uncertainty_threshold`: float (e.g., 5.0).
    *   `reference_potential`: "zbl" or "lj".

### 3.2. Hybrid Potential Implementation
To prevent unphysical atomic overlaps during exploration, we enforce a hybrid potential:
```lammps
pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace potential.yace ...
pair_coeff * * zbl ...
```
This ensures that even if the ACE potential (pace) behaves poorly at short range, the ZBL potential will provide a repulsive wall.

### 3.3. OTF Halt Mechanism
*   **Watchdog**: Use `compute pace/gamma` in LAMMPS.
*   **Trigger**: `fix halt` command stops the simulation if `max_gamma` exceeds the threshold.
*   **Handling**: The Python wrapper detects the halt (via return code or log parsing) and extracts the problematic structure for labeling.

## 4. Implementation Approach

1.  **LAMMPS Wrapper**: Implement `LammpsDynamics`. Construct `in.lammps` dynamically based on `DynamicsConfig`.
2.  **OTF Logic**: Implement the loop in `Orchestrator` (or `otf.py` helper) to handle the `Halt -> Select -> Oracle` transition.
3.  **Structure Extraction**: When halted, parse the dump file to get the snapshot with the highest uncertainty.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_dynamics.py` (Mocked)**:
    *   Verify `in.lammps` generation contains correct `pair_style` and `fix halt` commands.
*   **Mocked Execution**: Simulate a LAMMPS run that returns a "Halt" status and a dummy dump file.

### 5.2. Integration Testing
*   **Real MD (Requires LAMMPS with USER-PACE)**:
    *   Run a short MD (100 steps) with a trained potential.
    *   Verify it runs without crashing.
    *   Ideally, artificially induce high gamma (e.g., by distorting the structure) and verify `fix halt` works.
