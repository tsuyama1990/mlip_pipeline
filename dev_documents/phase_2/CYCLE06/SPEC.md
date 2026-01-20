# Cycle 06 Specification: Scalable Inference Engine (Part 1)

## 1. Summary

Cycle 06 initiates **Module E: Scalable Inference Engine**. Having trained a potential (Cycle 05), we must now use it. The purpose of this module is to run **Molecular Dynamics (MD)** simulations to explore the phase space of the material.

However, we are not just running MD for the sake of properties. We are running it to **stress-test** the potential. We use the **Extrapolation Grade ($\gamma$)** metric provided by the ACE potential to monitor "confidence" in real-time. If the potential encounters a configuration where $\gamma$ exceeds a safety threshold, it means the simulation has wandered into a region where the potential is untrained (and likely wrong). The engine must detect this, stop (or log), and flag the structure for re-training. This is the "Active Learning" part of the loop.

By the end of this cycle, we will be able to launch LAMMPS simulations that "self-diagnose" their reliability and report back interesting failures.

## 2. System Architecture

New components in `src/inference`.

```ascii
mlip_autopipec/
├── src/
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── **lammps_runner.py** # Wrapper for LAMMPS execution
│   │   ├── **uq.py**            # Uncertainty Quantification logic
│   │   ├── **inputs.py**        # LAMMPS script generator
│   │   └── **analysis.py**      # Basic property extraction (RDF, Diffusion)
│   └── config/
│       └── models.py            # Updated with InferenceConfig
├── tests/
│   └── inference/
│       ├── **test_lammps.py**
│       ├── **test_uq.py**
│       └── **test_analysis.py**
```

### Key Components

1.  **`LammpsRunner`**: Orchestrates the execution of `lmp_serial` or `lmp_mpi`. It handles the conversion of ASE atoms to LAMMPS data format and generates the input script (`in.lammps`). It manages the run directory and cleanup.
2.  **`ScriptGenerator`**: Creates the LAMMPS commands. Crucially, it includes the `pair_style pace` command and the `compute` command to calculate the extrapolation grade $\gamma$.
3.  **`UncertaintyChecker`**: Analyzes the output. It reads the LAMMPS dump file. If `dump_modify` was used to only output high-$\gamma$ frames, this class simply checks if the file is non-empty and loads the structures using `ase.io.read`.

## 3. Design Architecture

### Domain Concepts

**Extrapolation Grade ($\gamma$)**:
In ACE potentials, $\gamma$ measures how far a local atomic environment is from the training set in the descriptor space (based on D-optimality).
-   $\gamma \approx 1$: Well-interpolated. Safe.
-   $\gamma > 5$: Extrapolating. Warning.
-   $\gamma > 10$: Dangerous. The forces are likely garbage.

**The "Miner" Strategy**:
We treat the MD engine as a "Data Miner".
1.  Launch MD at high temperature (e.g., 2000K) or high pressure to force exploration.
2.  Configure LAMMPS to monitor $\gamma$.
3.  `compute max_gamma all reduce max c_pace[1]`
4.  `dump_modify my_dump thresh c_max_gamma > 5.0`
5.  If any atoms exceed the threshold, they are written to disk.
6.  We harvest these structures.

### Data Models

```python
class InferenceConfig(BaseModel):
    temperature: float
    pressure: float = 0.0 # 0 for NVT
    timestep: float = 0.001 # ps
    steps: int = 10000
    ensemble: Literal["nvt", "npt"] = "nvt"
    uq_threshold: float = 5.0
    sampling_interval: int = 100
    potential_path: Path

class InferenceResult(BaseModel):
    succeeded: bool
    final_structure: Path
    uncertain_structures: List[Path] # Paths to extracted frames
    max_gamma_observed: float
```

## 4. Implementation Approach

1.  **Step 1: LAMMPS Input (`inputs.py`)**:
    -   Implement `ScriptGenerator.generate()`.
    -   Must support `pair_style pace`.
    -   Must add `compute max_gamma all reduce max c_pace[1]` (assuming `c_pace` outputs gamma).
    -   Must add `run` command.
    -   Must add `dump` commands with thresholds.
2.  **Step 2: Runner (`lammps_runner.py`)**:
    -   Implement `LammpsRunner.run(atoms, potential_path)`.
    -   Write `data.lammps` (using `ase.io.write(..., format='lammps-data')`).
    -   Write `in.lammps`.
    -   Execute `subprocess.run`.
3.  **Step 3: UQ Handling (`uq.py`)**:
    -   Implement `UncertaintyChecker.parse_dump()`.
    -   Return a list of `Atoms` that were flagged.
    -   Assign metadata `src_md_step` to each atom.

## 5. Test Strategy

### Unit Testing Approach (Min 300 words)
-   **Script Verification**: We will generate LAMMPS scripts for various ensembles (NVT, NPT). We will regex-search the strings to ensure:
    -   `pair_style pace` is present.
    -   The potential file path is absolute and correct.
    -   The `thermo` output includes the Max Gamma variable.
    -   The `dump_modify` threshold matches the config.
-   **Data Conversion**: We will take a complex `Atoms` object (multi-species, Triclinic cell) and write it to LAMMPS data format. We will read it back and assert that positions and cell vectors are preserved (checking handling of tilt factors). We will check that atomic types (1, 2) map correctly to chemical symbols (Al, Cu).
-   **UQ Logic**: We will create a dummy dump file that contains 5 frames. We will assume 3 of them have atoms tagged with high gamma (simulated by custom columns). We will verify that `UncertaintyChecker` correctly identifies these 3 frames and ignores the others.

### Integration Testing Approach (Min 300 words)
-   **Mock LAMMPS**: We cannot assume LAMMPS is installed.
    -   **Scenario 1 (Stable)**: Mock `subprocess` to write a log file showing steady temperature and Max Gamma = 1.2. Verify `LammpsRunner` returns "Stable".
    -   **Scenario 2 (Unstable)**: Mock `subprocess` to write a dump file containing atoms. Verify `LammpsRunner` returns the parsed atoms as "Candidates".
    -   **Scenario 3 (Crash)**: Mock `subprocess` to return exit code 1. Verify `LammpsRunner` raises an appropriate exception but captures the `log.lammps` for debugging.
-   **Property Extraction**: We will mock a log file with 1000 steps of thermodynamic data (Temp, Press, Vol). We will use `AnalysisUtils` to parse this log and compute the average Temperature and its standard deviation. We assert the parser is robust to different LAMMPS log formats.
