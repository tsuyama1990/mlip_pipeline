# Cycle 05 Specification: Dynamics Engine & On-the-Fly Learning

## 1. Summary
Cycle 05 implements the "Dynamics Engine", enabling molecular dynamics (MD) simulations powered by the trained ACE potentials. Crucially, this cycle closes the Active Learning loop by implementing "On-the-Fly" (OTF) monitoring. The system automatically detects when the simulation enters a region of high uncertainty (high extrapolation grade $\gamma$), halts the simulation, and extracts the problematic structure for retraining.

Key features:
1.  **LAMMPS Integration**: A flexible interface to generate LAMMPS input scripts and execute simulations.
2.  **Hybrid Potential Logic**: Implementation of the safety mechanism that superimposes a physical baseline (LJ/ZBL) onto the ML potential (`pair_style hybrid/overlay`), preventing unphysical behavior in unknown regions.
3.  **Halt & Diagnose**: A watchdog mechanism (`fix halt`) that monitors $\gamma$ in real-time. If it exceeds a threshold, the simulation stops, and the system extracts the "bad" structure.
4.  **Orchestrator Logic**: The central brain that connects all modules (Generator -> Dynamics -> Oracle -> Trainer) into an autonomous loop.

## 2. System Architecture

The file structure expands `src/pyacemaker/dynamics` and `src/pyacemaker/core`. **Bold files** are new.

```text
src/
└── pyacemaker/
    ├── core/
    │   ├── **config.py**       # Updated with DynamicsConfig
    │   └── **orchestrator.py** # The Main Loop Logic
    └── **dynamics/**
        ├── **__init__.py**
        ├── **md.py**           # LAMMPS Interface & Halt Logic
        └── **potential.py**    # Hybrid Potential Helper
```

### File Details
-   `src/pyacemaker/dynamics/md.py`: Contains `MDInterface`. It writes `in.lammps`, runs `lmp`, and parses output.
-   `src/pyacemaker/dynamics/potential.py`: Helper class to generate the correct `pair_style` and `pair_coeff` commands for hybrid potentials (ACE + ZBL/LJ).
-   `src/pyacemaker/core/orchestrator.py`: Implements the `Orchestrator` class. It manages the state of the active learning cycle (iteration number, current potential, dataset size).
-   `src/pyacemaker/core/config.py`: Expanded to include `DynamicsConfig` (timestep, temperature, halt_threshold).

## 3. Design Architecture

### 3.1. Dynamics Configuration
```python
class DynamicsConfig(BaseModel):
    timestep: float = 0.001 # ps
    temperature: float = 300.0
    pressure: float = 0.0
    n_steps: int = 100000
    halt_threshold: float = 5.0 # Max gamma before halting
    hybrid_baseline: str = "zbl" # "lj" or "zbl"
```

### 3.2. Halt Info Model
```python
class HaltInfo(BaseModel):
    halted: bool
    step: int
    max_gamma: float
    structure: Optional[Atoms] # The snapshot where halt occurred
```

### 3.3. Orchestrator State Machine
```python
class Orchestrator:
    def run_cycle(self):
        # 1. Exploration (MD)
        halt_info = self.dynamics.run(self.current_potential)

        if halt_info.halted:
            # 2. Selection (Active Learning)
            candidates = self.generator.generate_local(halt_info.structure)
            selected = self.trainer.select_active_set(candidates)

            # 3. Labeling (Oracle)
            new_data = self.oracle.compute(selected)
            self.dataset.add(new_data)

            # 4. Training (Trainer)
            self.current_potential = self.trainer.train(self.dataset)
```

## 4. Implementation Approach

### Step 1: Update Configuration
-   Modify `src/pyacemaker/core/config.py` to add `DynamicsConfig`.

### Step 2: Hybrid Potential Helper
-   Implement `src/pyacemaker/dynamics/potential.py`.
-   `get_lammps_commands(potential_path, baseline_type)`: Returns a list of strings (e.g., `pair_style hybrid/overlay ...`).

### Step 3: LAMMPS Interface
-   Implement `src/pyacemaker/dynamics/md.py`.
-   `run_md(structure, potential, work_dir)`:
    -   Write `in.lammps` with `fix halt` command monitoring `c_pace_gamma`.
    -   Execute LAMMPS via `subprocess`.
    -   If exit code indicates halt (or log contains "Halt triggered"), parse the dump file.
-   `extract_bad_structure(dump_file)`: Read the last frame from the dump.

### Step 4: Orchestrator Integration
-   Implement `src/pyacemaker/core/orchestrator.py`.
-   Connect `MDInterface` with `Oracle` and `Trainer`.
-   Implement the "Halt & Diagnose" loop logic.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Potential Helper (`tests/dynamics/test_potential.py`)**:
    -   Verify that `zbl` baseline generates `pair_style hybrid/overlay pace zbl ...`.
    -   Verify that `lj` baseline generates correct parameters.
-   **MD Interface (`tests/dynamics/test_md.py`)**:
    -   Mock `subprocess.run`.
    -   Verify `in.lammps` content includes `fix halt` and `compute pace`.
    -   Simulate a halt scenario: Create a dummy dump file, verify `extract_bad_structure` returns the correct frame.

### 5.2. Integration Testing
-   **Orchestrator Loop (`tests/core/test_orchestrator.py`)**:
    -   Mock all subsystems (`Dynamics`, `Oracle`, `Trainer`).
    -   Simulate `Dynamics` returning `HaltInfo(halted=True)`.
    -   Verify that `Oracle.compute` and `Trainer.train` are called in sequence.
    -   Verify that iteration counter increments.
