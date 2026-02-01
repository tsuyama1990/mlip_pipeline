# Cycle 03: The Oracle (DFT Automation)

## 1. Summary

In Cycle 02, we established the ability to run classical simulations (MD). Now, in Cycle 03, we build the "Teacher" or "Oracle" of our active learning system: the Density Functional Theory (DFT) engine. This module is responsible for providing the ground truth labels (Energy, Forces, Virial Stress) that the machine learning model will attempt to mimic.

The challenge with DFT is its fragility. Unlike classical MD, which almost always runs to completion (even if the physics is wrong), DFT calculations often fail to converge electronically (SCF cycles), especially for the distorted, high-temperature, or defective structures generated during active learning exploration. A naive script that simply runs `pw.x` will fail 20-30% of the time, halting the entire autonomous pipeline.

Therefore, the core objective of this cycle is not just to run Quantum Espresso, but to implement a **Robust, Self-Healing Interface**. This "Recovery Handler" must detect specific error patterns (e.g., "convergence not achieved", "charge density negative") and automatically adjust physical parameters—such as increasing the electron temperature (smearing), reducing the mixing beta, or changing the diagonalization algorithm—to salvage the calculation.

We will also implement **Periodic Embedding**. Since we cannot afford to run DFT on thousands of atoms, we need a mechanism to take a cluster from an MD simulation, wrap it in a vacuum-padded periodic box, and run the calculation on this smaller "representative" system.

## 2. System Architecture

We introduce the `physics/dft` package.

### File Structure
Files to be created/modified are in **bold**.

```ascii
mlip_autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   ├── **calculation.py**      # DFT specific schemas
│       │   └── config.py               # Update with DFTConfig
│       ├── physics/
│       │   ├── dft/
│       │   │   ├── **__init__.py**
│       │   │   ├── **qe_runner.py**    # Quantum Espresso Wrapper
│       │   │   ├── **input_gen.py**    # Input file creation logic
│       │   │   ├── **parser.py**       # Output parsing
│       │   │   └── **recovery.py**     # Error handling logic
│       │   └── structure_gen/
│       │       └── **embedding.py**    # Cluster extraction logic
└── tests/
    └── physics/
        └── dft/
            ├── **test_input_gen.py**
            └── **test_recovery.py**
```

### Component Interaction

1.  **Orchestrator** identifies a `Structure` that needs labelling.
2.  **`EmbeddingHandler`** (Optional): If the structure is huge, it cuts a cluster and returns a smaller `Structure`.
3.  **`QERunner`** initiates a job.
4.  **`InputGenerator`** reads `DFTConfig` (pseudopotentials, cutoffs) and writes `pw.in`.
5.  **`QERunner`** executes `pw.x`.
6.  **`Parser`** checks the output.
    -   If **Converged**: Returns `DFTResult` (Energy, Forces, Stress).
    -   If **Failed**: Raises a specific `DFTError` (e.g., `SCFConvergenceError`).
7.  **`RecoveryHandler`** catches the error.
    -   It looks up a "Cure" (e.g., `mix_beta *= 0.5`).
    -   It modifies the params and triggers a re-run (recursion up to N times).

## 3. Design Architecture

### 3.1. Calculation Domain Model (`domain_models/calculation.py`)

-   **Class `DFTConfig`**:
    -   `command`: `str` (e.g., `mpirun -np 16 pw.x`).
    -   `pseudopotentials`: `Dict[str, Path]`.
    -   `ecutwfc`: `float` (Wavefunction cutoff).
    -   `kspacing`: `float` (Inverse k-point density, e.g., 0.04).

-   **Class `DFTResult`**:
    -   `energy`: `float` (eV).
    -   `forces`: `NDArray[(N, 3), float]` (eV/A).
    -   `stress`: `NDArray[(3, 3), float]` (eV/A^3, optional).
    -   `magmoms`: `Optional[NDArray]` (Magnetic moments).

-   **Exception `DFTError`**:
    -   Base class for `SCFError`, `MemoryError`, `WalltimeError`.

### 3.2. Recovery Logic (`physics/dft/recovery.py`)

This is a state machine or a rule-based engine.
-   **Logic**:
    ```python
    RULES = [
        (SCFConvergenceError, {"mixing_beta": 0.3}),  # Try softer mixing
        (SCFConvergenceError, {"smearing": "mv", "degauss": 0.02}), # Try hotter electrons
        (DiagonalizationError, {"diagonalization": "cg"}), # Change algo
    ]
    ```

### 3.3. Input Generation
-   **Key Constraint**: Must support `kspacing` logic. Instead of fixed K-points (e.g., 4x4x4), we calculate the grid based on the cell size: $N_i = \text{ceil}(2\pi / (L_i \times \text{kspacing}))$. This ensures consistent accuracy across different cell sizes (e.g., bulk vs. supercell).

## 4. Implementation Approach

### Step 1: Domain Models
-   Update `config.py` to include `dft` section.
-   Create `calculation.py`.

### Step 2: Input Generator
-   Implement `input_gen.py`.
-   Use `ase.io.write(format='espresso-in')` as a base, but wrap it to strictly control parameters like `tprnfor` (print forces) and `tstress` (print stress) which are required for MLIP training.

### Step 3: Parser
-   Implement `parser.py`.
-   Parse standard text output or XML output of QE.
-   Crucially, implement regex patterns to detect specific crash messages (e.g., "convergence not achieved").

### Step 4: Runner & Recovery
-   Implement `qe_runner.py`.
-   It should accept a `Structure` and `DFTConfig`.
-   Implement the retry loop:
    ```python
    for attempt in range(max_retries):
        try:
            run_process()
            check_convergence()
            return parse_results()
        except DFTError as e:
            params = recovery_handler.apply_fix(params, e)
    ```

## 5. Test Strategy

### 5.1. Unit Testing
-   **Input Generation**:
    -   Pass a `Structure` and check the generated string. Ensure `K_POINTS automatic` matches the `kspacing` formula.
    -   Ensure `tprnfor=.true.` is present.
-   **Recovery Logic**:
    -   Simulate an `SCFConvergenceError`. Call `apply_fix`. Assert that the returned parameters have a lower `mixing_beta`.

### 5.2. Integration Testing (Mocked)
-   **Mocking QE**:
    -   Create a mock `subprocess.run` that reads a pre-saved "failed" QE output file (text) and returns it as stdout.
    -   Assert that `QERunner` detects the failure and attempts a second run.
    -   For the second run, mock a "success" output.
    -   Assert that the final result is returned and `retries=1`.

### 5.3. Real DFT Test (Optional/Local)
-   If `pw.x` is available, run a static calculation on a single H2 molecule.
-   Check that Forces are close to zero for equilibrium bond length.
