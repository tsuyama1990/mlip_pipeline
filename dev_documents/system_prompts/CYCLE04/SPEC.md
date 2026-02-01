# Specification: Cycle 04 - The Executor (Dynamics & Active Learning)

## 1. Summary

Cycle 04 connects the brain (Orchestrator) to the muscle (Dynamics Engine). In this cycle, we implement the ability to run **Molecular Dynamics (MD)** simulations using **LAMMPS**, the industry-standard code for large-scale materials modeling.

This cycle is where the "Active" in "Active Learning" comes alive. We are not just running MD; we are implementing **On-the-Fly (OTF) Uncertainty Quantification**. The system will configure LAMMPS to monitor the extrapolation grade ($\gamma$) of the ACE potential at every timestep. If $\gamma$ exceeds a safety threshold, the simulation will self-terminate ("Halt"). The Orchestrator will then diagnose the crash, extract the high-uncertainty configuration, and send it to the Oracle (Cycle 02) for labeling.

Additionally, this cycle implements the **Hybrid Potential** architecture. To prevent physical disasters (atoms overlapping) in the early stages of learning, we will programmatically mix the machine learning potential with a physics-based baseline (ZBL or Lennard-Jones) using LAMMPS's `pair_style hybrid/overlay`.

## 2. System Architecture

### 2.1 File Structure

Files to be created or modified (bold):

```
mlip-autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── config/
│       │   └── **config_model.py**         # Update with LammpsConfig
│       ├── physics/
│       │   └── dynamics/
│       │       ├── __init__.py
│       │       ├── **lammps_runner.py**    # The Driver
│       │       ├── **input_generator.py**  # Creates in.lammps
│       │       └── **log_parser.py**       # Reads lammps.log
│       └── orchestration/
│           └── **otf_loop.py**             # Logic to handle Halt events
├── tests/
│   ├── unit/
│   │   └── **test_lammps_input.py**
│   └── integration/
│       └── **test_otf_cycle.py**
└── config.yaml
```

### 2.2 Component Blueprints

#### `src/mlip_autopipec/physics/dynamics/lammps_runner.py`

```python
class LammpsRunner:
    def run(self, structure: Atoms, potential: Potential, settings: dict) -> MDResult:
        """
        1. Generates in.lammps with pair_style hybrid/overlay.
        2. Runs 'lmp -in in.lammps'.
        3. Monitors output for 'Fix halt' trigger.
        4. Returns trajectory and status (COMPLETED or HALTED).
        """
        ...
```

#### `src/mlip_autopipec/physics/dynamics/input_generator.py`

```python
def generate_hybrid_pair_style(elements: list[str]) -> str:
    """
    Returns the LAMMPS commands for safety.
    Example:
    pair_style hybrid/overlay pace zbl 1.0 2.0
    pair_coeff * * pace potential.yace ...
    pair_coeff * * zbl 14 14 ...
    """
    ...
```

## 3. Design Architecture

### 3.1 The "Watchdog" Pattern (Fix Halt)
Instead of running MD for 1ns and analyzing it afterwards, we use LAMMPS's internal `fix halt` command.
*   **Command**: `fix watchdog all halt 10 v_max_gamma > 5.0 error hard`
*   **Behavior**: LAMMPS checks the variable `max_gamma` every 10 steps. If it exceeds 5.0, it kills the process with a specific error message.
*   **Benefit**: We stop *exactly* when the potential enters unknown territory. We don't waste compute time simulating "garbage" after the potential has failed.

### 3.2 Hybrid Overlay (Safety Net)
The potential energy is defined as $E_{total} = E_{ACE} + E_{ZBL}$.
*   **ZBL**: A steep repulsive wall at short distances ($r < 1.0 \AA$).
*   **ACE**: The machine learned many-body interaction.
*   **Design**: The `InputGenerator` must automatically look up the atomic numbers of the species and generate the correct ZBL parameters. This must happen transparently to the user.

## 4. Implementation Approach

### Step 1: Input Generation
1.  Implement `InputGenerator`. It needs to take an `Atoms` object and write a `data.lammps` file.
2.  Implement the `hybrid/overlay` logic. It needs to write the `pair_style` and `pair_coeff` lines.

### Step 2: LAMMPS Execution
1.  Implement `LammpsRunner`. Use `subprocess.Popen` to stream stdout if possible, or wait for completion and read log.
2.  Handle the `ACE_EPOCH_Warning` or similar messages that indicate extrapolation.

### Step 3: Log Parsing
1.  Implement `LogParser`. It needs to read `lammps.log` and extract thermodynamic data (T, P, V, Epot).
2.  **Critical**: It must detect if the run finished normally ("Loop time of ...") or if it crashed/halted. If halted, it must find the timestep where it happened.

### Step 4: OTF Loop Integration
1.  Update `Orchestrator` to use `LammpsRunner`.
2.  Implement the logic:
    *   `result = runner.run(...)`
    *   `if result.status == HALTED:`
        *   `bad_structure = runner.get_snapshot(result.halt_step)`
        *   `candidates = embedder.extract_periodic_box(bad_structure)`
        *   `oracle.compute(candidates)`

## 5. Test Strategy

### 5.1 Unit Testing Approach (Min 300 words)
*   **Input Validity**: We will generate `in.lammps` for a binary alloy (e.g., Ti-O). We will regex-search the string to ensure:
    *   `pair_style hybrid/overlay` exists.
    *   `pair_coeff * * pace` exists.
    *   `fix watchdog` is defined with the correct threshold.
*   **Log Parsing**: We will create dummy log files representing different scenarios:
    *   **Normal**: Contains standard thermo output and "Loop time".
    *   **Halt**: Contains "ERROR: Fix halt condition met".
    *   **Crash**: Contains "Segmentation fault" or empty file.
    *   We assert that `LogParser` returns the correct Enum status for each file.

### 5.2 Integration Testing Approach (Min 300 words)
*   **Mock-LAMMPS**: Since we cannot guarantee `lmp` and `potential.yace` are present in CI, we will create a `mock_lammps.py` script.
    *   This script accepts `-in in.lammps`.
    *   It reads the input file.
    *   It prints standard LAMMPS-like thermo output to stdout.
    *   If the input file contains a specific comment `# TEST_TRIGGER_HALT`, the mock script will print "ERROR: Fix halt condition met" after 50 steps.
*   **Cycle Test**: We will configure the Orchestrator to use this mock script. we will verify that when the halt is triggered, the Orchestrator correctly identifies the "Halted" state and attempts to retrieve the structure (which the mock script should also generate as `dump.lammpstrj`).
