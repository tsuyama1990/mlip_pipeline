# Cycle 03 Specification: Dynamics Engine & Uncertainty

## 1. Summary
Cycle 03 introduces the **Dynamics Engine**, enabling Molecular Dynamics (MD) simulations via LAMMPS. The critical innovation here is the **Hybrid Potential** strategy, which combines ACE with a physical baseline (ZBL/LJ) to prevent crashes in extrapolation regions. We also implement the **Uncertainty Monitor**, which uses the `fix halt` command in LAMMPS to interrupt simulations when the extrapolation grade ($\gamma$) exceeds a safety threshold.

## 2. System Architecture

### 2.1. File Structure
```text
src/mlip_autopipec/
├── dynamics/                       # [CREATE]
│   ├── __init__.py
│   ├── lammps_runner.py            # [CREATE] LAMMPS wrapper
│   ├── lammps_input.py             # [CREATE] Input file generator
│   └── log_parser.py               # [CREATE] Log analysis
└── config/
    └── dynamics_config.py          # [CREATE] MD settings
```

### 2.2. Component Interaction
- **`LammpsRunner`**: Orchestrates the MD run. It prepares the directory, writes `in.lammps` and `data.lammps`, and executes the binary.
- **`LammpsInputWriter`**: Generates the complex `pair_style hybrid/overlay` commands. It ensures the ACE potential (`pace`) and the baseline (`zbl` or `lj/cut`) are correctly superimposed.
- **`LogParser`**: Reads `lammps.log` to determine if the run finished normally or was halted by the watchdog. It extracts the maximum $\gamma$ value encountered.

## 3. Design Architecture

### 3.1. Hybrid Potential Logic
- **Objective**: $E_{total} = E_{ACE} + E_{Baseline}$
- **Implementation**:
    ```lammps
    pair_style hybrid/overlay pace zbl 1.0 2.0
    pair_coeff * * pace potential.yace Element
    pair_coeff * * zbl 14 14
    ```
- **Constraint**: The `zbl` parameters (atomic numbers) must match the elements in the system.

### 3.2. Uncertainty Watchdog
- **Objective**: Stop simulation if $\gamma > \gamma_{thresh}$.
- **Implementation**:
    ```lammps
    compute pace_gamma all pace potential.yace gamma_mode=1
    variable max_gamma equal max(c_pace_gamma)
    fix watchdog all halt 10 v_max_gamma > 5.0 error hard
    ```
- **Handling**: The Python wrapper must catch the "error hard" exit code (usually non-zero) and interpret it as a "Discovery" event, not a system failure.

## 4. Implementation Approach

1.  **Input Writer**: Create a class that accepts `Atoms`, `potential_path`, and `md_params`. It should render a Jinja2 template for `in.lammps`.
2.  **Runner**: Implement `run_md()`.
    -   Setup working directory.
    -   Run LAMMPS (using `lammps` python module if available, or `subprocess`).
    -   Return a status object: `{'halted': bool, 'max_gamma': float, 'final_structure': Atoms}`.
3.  **Log Parser**: Regex-based parsing of the log file to find the "fix halt" message.

## 5. Test Strategy

### 5.1. Unit Testing
- **Input Generation**: specific test to ensure `pair_style hybrid/overlay` is correctly formatted in the generated string.
- **Log Parsing**: Feed a dummy log file containing a halt message and verify the parser returns `halted=True`.

### 5.2. Integration Testing
- **Dry Run**: If a LAMMPS binary with USER-PACE is available, run a 10-step MD.
- **Mock Run**: Create a mock script that acts as LAMMPS, writing a log file that says "Fix halt condition met". Verify the runner correctly identifies this as a valid halt event.
