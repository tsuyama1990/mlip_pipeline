# Cycle 01: Core Framework & DFT Factory

## 1. Summary

Cycle 01 forms the bedrock of the **MLIP-AutoPipe** initiative. In the realm of automated machine learning for materials science, the integrity of the data pipeline is paramount. The "Garbage In, Garbage Out" principle applies strictly: if the First-Principles (DFT) training data is noisy, unconverged, or inconsistent, the resulting Machine Learning Interatomic Potential (MLIP) will be physically invalid, leading to catastrophic simulation failures in later stages. Therefore, this cycle is dedicated not to ML, but to the rigorous engineering of the data generation factory.

The primary deliverable of this cycle is **Module C: Automated DFT Factory**. We aim to construct a highly resilient, autonomous system capable of executing high-throughput Density Functional Theory calculations using **Quantum Espresso (QE)**. Unlike manual workflows where a researcher hand-tunes parameters for each job, our factory must implement a robust "Static Calculation Protocol" that guarantees consistency across thousands of calculations. This involves automating the selection of pseudopotentials (via the SSSP library), the generation of K-point grids (using linear density heuristics), and the handling of spin polarization for magnetic materials.

Beyond the physics engine, this cycle establishes the software engineering foundation for the entire project. We will implement a Type-Safe configuration system using **Pydantic**, ensuring that invalid user inputs are caught at startup rather than crashing the pipeline days later. We will also implement a persistent storage layer using the **ASE Database**, providing a structured, queryable repository for the massive datasets we will generate. By the end of this cycle, we will have a functional "Black Box" that accepts an atomic structure and reliably returns its Quantum Mechanical energy, forces, and virial stress, laying the groundwork for the active learning loop.

## 2. System Architecture

### 2.1. Code Blueprint and File Structure

The architecture for this cycle is designed to isolate the "Business Logic" (DFT execution) from the "Infrastructure" (Config, Database). This separation of concerns ensures that we can swap out the DFT engine (e.g., to VASP) or the Database (e.g., to MongoDB) in the future without rewriting the entire codebase.

The following file structure will be implemented. Files in **bold** are new or significantly modified in this cycle.

```
mlip_autopipec/
├── **__init__.py**                 # Package initialization
├── **app.py**                      # Main CLI entry point for integration testing
├── core/
│   ├── **__init__.py**
│   ├── **config.py**               # Pydantic schemas for Global and DFT configuration
│   ├── **database.py**             # Database abstraction layer (ASE db wrapper)
│   ├── **logging.py**              # Centralized logging configuration (Rich)
│   └── **exceptions.py**           # Custom exception hierarchy (DFTError, ConfigError)
└── dft/
    ├── **__init__.py**
    ├── **qe_runner.py**            # The core Quantum Espresso execution engine
    ├── **utils.py**                # Helpers: K-points, Magnetism detection, SSSP logic
    ├── **input_gen.py**            # Logic for generating valid pw.x input text
    └── **parsers.py**              # Logic for parsing pw.x output logs (XML/Text)
```

### 2.2. Component Interaction and Data Flow

The data flow in Cycle 01 is linear and synchronous, prioritizing reliability over throughput.

1.  **Configuration Loading**:
    The lifecycle begins when `core.config.load_config()` reads the user's YAML file. It instantiates the `DFTConfig` Pydantic model. This model enforces constraints (e.g., `ecutwfc > 0`, `pseudopotential_dir` exists). If validation fails, the program exits immediately with a clear error message, preventing "Zombie" runs.

2.  **Input Preparation (The `dft` Package)**:
    The `QERunner` class receives a candidate `ase.Atoms` object. It does not blindly run `pw.x`. Instead, it orchestrates a setup phase:
    -   **Pseudopotential Mapping**: It queries `dft.utils.get_sssp_pseudopotentials()` to map each element (e.g., "Fe") to its corresponding UPF file.
    -   **K-Point Generation**: It calls `dft.utils.get_kpoints(atoms, density=0.15)` to calculate a Monkhorst-Pack grid that ensures uniform sampling of the Brillouin zone, regardless of cell size.
    -   **Magnetism**: It checks `dft.utils.is_magnetic(atoms)`. If magnetic elements (Fe, Ni, Co) are present, it automatically sets `nspin=2` and initializes random magnetic moments to break symmetry.
    -   **Input Writing**: Finally, `dft.input_gen.write_pw_input()` synthesizes these parameters into a valid text string formatted for Quantum Espresso.

3.  **Execution (The Subprocess Boundary)**:
    The `QERunner` invokes the external `pw.x` binary using `subprocess.run`. Crucially, this execution is wrapped in a `try/except` block with a timeout. This acts as a "Dead Man's Switch": if `pw.x` hangs (a common issue with MPI errors), the runner kills the process after a predefined limit (e.g., 4 hours), preventing compute node locking.

4.  **Parsing and Verification**:
    Upon process completion, `dft.parsers.parse_pw_output()` reads the standard output (or XML file). It looks for specific "success markers" (e.g., "JOB DONE"). If these are missing, it raises a `DFTRuntimeError`. If successful, it extracts the total energy (eV), atomic forces (eV/A), and stress tensor. It performs a sanity check: if `forces` contains `NaN` or `Inf`, the result is discarded.

5.  **Persistence (The `core` Package)**:
    The valid `Atoms` object is passed to `core.database.DatabaseManager`. This class manages the SQLite connection. It serializes the atoms and their properties, tags them with metadata (e.g., `config_type="dft_scf"`, `calculator="qe-7.2"`), and commits the transaction.

## 3. Design Architecture

### 3.1. Configuration Models (`core/config.py`)

We leverage **Pydantic V2** to define the system's "Contract". This ensures that the code never has to deal with untyped dictionaries or missing keys.

-   **`DFTConfig`**:
    -   `code`: `Literal["quantum_espresso"]`. Hardcoded for now but extensible.
    -   `command`: `str`. The exact command string (e.g., `mpirun -np 32 pw.x -in pw.in > pw.out`).
    -   `pseudopotential_dir`: `FilePath`. Must point to a valid directory.
    -   `scf_convergence_threshold`: `float` (default `1e-6` Ry). Stricter than default to ensure accurate forces.
    -   `mixing_beta`: `float` (default `0.7`). The mixing factor for electron density.
    -   `smearing`: `str` (default `mv`). The smearing scheme (Marzari-Vanderbilt) is robust for both metals and insulators.

-   **`GlobalConfig`**:
    -   `project_name`: `str`. Used for folder naming.
    -   `database_path`: `FilePath`. Location of the `.db` file.
    -   `logging_level`: `Literal["DEBUG", "INFO", "WARNING"]`.

### 3.2. The Database Manager (`core/database.py`)

This class wraps the `ase.db` library, providing a domain-specific API that hides the SQL details.

-   **Class `DatabaseManager`**:
    -   **`__init__(self, db_path: Path)`**: Establishes the connection. Creates the file if it doesn't exist.
    -   **`count(self, **kwargs) -> int`**: Returns the number of rows matching the query.
    -   **`add_calculation(self, atoms: Atoms, metadata: Dict) -> int`**:
        -   *Invariant*: The `atoms` object must have a calculator attached (or results dict).
        -   *Invariant*: `energy` and `forces` must be present.
        -   *Logic*: Flattens the metadata dictionary and stores it as key-value pairs in the DB columns. Returns the Row ID.
    -   **`get_pending_calculations(self) -> List[Atoms]`**: Retrieves entries flagged for computation (future use).

### 3.3. The QE Runner (`dft/qe_runner.py`)

This is the core implementation of the "Automated Factory".

-   **Class `QERunner`**:
    -   **`__init__(self, config: DFTConfig)`**: Stores the configuration.
    -   **`run_static_calculation(self, atoms: Atoms, run_dir: Path) -> DFTResult`**:
        -   *Input*: An atomic structure and a scratch directory.
        -   *Output*: A `DFTResult` object (defined in `core/models.py`) containing the physical properties.
        -   *Logic*:
            1.  **Sanitize**: Ensure atoms object fits in the simulation box.
            2.  **Generate**: Call `input_gen` to create `pw.in`.
            3.  **Execute**: Run the binary.
            4.  **Parse**: Call `parsers` to read `pw.out`.
            5.  **Clean**: Remove bulky temporary files (`.wfc`, `.hub`, `.mix`) to save disk space, keeping only the input/output text files.

### 3.4. Helper Modules (`dft/utils.py`, `dft/input_gen.py`)

-   **`get_kpoints(atoms, density)`**:
    -   Calculates the reciprocal lattice vectors $b_1, b_2, b_3$.
    -   Determines integers $k_i = \max(1, \text{round}(|b_i| \times \text{density}))$.
    -   Returns the grid `[k1, k2, k3]`.

-   **`write_pw_input(atoms, parameters, ...)`**:
    -   Uses `ase.io.write(format='espresso-in')` but augments it with custom flags (e.g., `tprnfor`, `tstress`) that ASE might not set by default for all versions.

## 4. Implementation Approach

We will follow a component-wise implementation strategy, building from the bottom up.

1.  **Phase 1: Configuration & Infrastructure**
    -   Create the `core` package.
    -   Implement `config.py` with Pydantic. Write unit tests to verify validation rules (e.g., passing a string for `mixing_beta` should fail).
    -   Implement `logging.py` to set up structured logging.

2.  **Phase 2: Database Layer**
    -   Implement `database.py`.
    -   Create a temporary SQLite DB in tests to verify read/write operations. Ensure that numpy arrays (forces) are serialized correctly.

3.  **Phase 3: DFT Logic (The "Dry Run")**
    -   Implement `dft/utils.py` and `dft/input_gen.py`.
    -   Write tests that generate input files for various systems (Al, FeO, H2O) and verify the text content matches our expectations (e.g., "Does Fe have nspin=2?").

4.  **Phase 4: Execution Engine (The "Wet Run")**
    -   Implement `QERunner`.
    -   **Mock Strategy**: Since we cannot guarantee `pw.x` availability in all environments, we will create a `MockQERunner` for testing. This class mimics the interface but writes a pre-canned `pw.out` file instead of running MPI.
    -   **Integration**: Write a script `app.py` that ties Config -> Runner -> DB together.

## 5. Test Strategy

### 5.1. Unit Testing (Pytest)

-   **`tests/core/test_config.py`**:
    -   Verify defaults: `kpoints_density` should be 0.15.
    -   Verify error handling: `scf_convergence_threshold=0` should raise ValidationError.
-   **`tests/dft/test_utils.py`**:
    -   **K-points**: Create a $10 \times 10 \times 10$ cell. Grid should be `[1, 1, 1]`. Create a $2 \times 2 \times 2$ cell. Grid should be `[5, 5, 5]`.
    -   **Magnetism**: Pass `Atoms('Fe')`. Assert `is_magnetic` returns True. Pass `Atoms('Si')`. Assert False.
-   **`tests/dft/test_input_gen.py`**:
    -   Generate input for a system. Check that `calculation='scf'` is present. Check that `disk_io='low'` is present.

### 5.2. Integration Testing

-   **`tests/integration/test_qe_mock.py`**:
    -   Use `pytest-mock` to intercept `subprocess.run`.
    -   Make the mock write a valid `pw.out` file containing:
        ```text
        !    total energy              =     -100.00 Ry
        Forces acting on atoms (cartesian axes, Ry/au):
        atom    1 type  1   force =     0.0    0.0    0.0
        JOB DONE.
        ```
    -   Run `QERunner.run_static_calculation`.
    -   Assert the returned object has `energy = -1360.5` eV.

### 5.3. Exception Testing

-   **Broken Input**: Pass an `Atoms` object with overlapping atoms (distance 0.1A). The runner should generate the input, but the Mock (simulating QE) should return an error or the parser should flag high forces.
-   **Convergence Failure**: Create a mock output that ends without "JOB DONE". Assert `QERunner` raises `DFTConvergenceError`.
