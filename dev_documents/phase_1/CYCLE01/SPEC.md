# CYCLE01: The Foundation - Automated DFT Factory (SPEC.md)

## 1. Summary

This document provides the detailed technical specification for Cycle 1 of the MLIP-AutoPipe project. The singular focus of this inaugural cycle is to construct the project's cornerstone: a robust, autonomous, and fault-tolerant **Automated DFT Factory**. This component is arguably the most critical piece of the entire system, as the quality and reliability of all subsequent machine learning processes depend entirely on its ability to consistently and accurately perform quantum mechanical calculations. The primary goal is to abstract away the immense complexity of setting up and running Density Functional Theory (DFT) calculations, providing a simple, high-level interface to the rest of the application.

The core deliverable for this cycle is a Python module, specifically a `DFTFactory` class, that can accept a standard `ase.Atoms` object—a common representation of an atomic structure—and return its calculated energy, atomic forces, and virial stress. This process must be fully automated, encapsulating expert knowledge that is typically required from a human researcher. This includes intelligent, heuristic-based determination of crucial DFT parameters such as plane-wave cutoff energies, k-point mesh densities, and magnetic moments. By codifying this domain knowledge, the system eliminates a significant barrier to entry and a common source of error.

Furthermore, this cycle places a heavy emphasis on resilience. DFT calculations, especially within an automated workflow that explores diverse and often distorted structures, are prone to convergence failures. A key feature of the `DFTFactory` will be its sophisticated **auto-recovery mechanism**. It will be designed to parse the output of a failed calculation, diagnose the likely cause of the error, and automatically adjust specific parameters—such as the electronic structure mixing scheme or smearing temperature—before resubmitting the job. This capability is non-negotiable for achieving the "Zero-Human" objective, ensuring the workflow can proceed without manual intervention even when faced with computational difficulties.

Finally, Cycle 1 will also establish the foundational data persistence layer. All successful DFT results will be stored in a structured format within an ASE-compatible database (e.g., SQLite). This creates a centralised, queryable repository of training data that will be consumed by the machine learning components in subsequent cycles. By the end of this cycle, the project will have a command-line-callable tool that can reliably take a structure and add its DFT properties to a database, forming the bedrock upon which the entire active learning architecture will be built.

## 2. System Architecture

The architecture for Cycle 1 is focused on creating a self-contained, testable, and reusable module for DFT calculations. It introduces the `mlip_autopipec/modules/dft.py` file, which will contain the main logic, and establishes the data persistence layer in `mlip_autopipec/utils/ase_utils.py`.

**File Structure for Cycle 1:**

The following ASCII tree shows the files that will be created or modified in this cycle. New files are marked in **bold**.

```
mlip-autopipe/
├── dev_documents/
│   └── system_prompts/
│       └── CYCLE01/
│           ├── **SPEC.md**
│           └── **UAT.md**
├── mlip_autopipec/
│   ├── __init__.py
│   ├── modules/
│   │   ├── __init__.py
│   │   └── **dft.py**          # Core DFT Factory implementation
│   └── utils/
│       ├── __init__.py
│       └── **ase_utils.py**    # Database interaction helpers
├── tests/
│   └── modules/
│       └── **test_dft.py**     # Unit and integration tests for dft.py
└── pyproject.toml
```

**Component Blueprint: `modules/dft.py`**

This file will house the `DFTFactory` class, the primary interface for running calculations.

-   **`DFTFactory` class:**
    -   **`__init__(self, dft_config)`**: The constructor will take a Pydantic configuration object that specifies settings like the path to the DFT executable (`qeVersions.x`), pseudopotentials, and resource limits (e.g., number of cores). This use of dependency injection makes the class highly configurable and testable.
    -   **`run(self, atoms: ase.Atoms) -> DFTResult`**: This is the main public method. It orchestrates the entire calculation process for a given `ase.Atoms` object. It will internally call a series of private helper methods to manage the workflow: preparation, execution, parsing, and error handling. It will return a structured Pydantic `DFTResult` object on success or raise a specific exception (e.g., `DFTCalculationError`) if the calculation fails after all retry attempts.
    -   **`_prepare_input_files(self, atoms: ase.Atoms, params: dict) -> str`**: This method will take the atoms object and a dictionary of calculation parameters and generate the input file for Quantum Espresso. It will use ASE's built-in calculators as a foundation but will add the specific heuristic-derived settings. It returns the path to the generated input file.
    -   **`_execute_dft(self, input_path: str) -> subprocess.CompletedProcess`**: This method is responsible for invoking the external DFT code as a subprocess. It will use `subprocess.run`, ensuring that it captures `stdout` and `stderr` and waits for the process to complete. It will not use `shell=True` to prevent command injection vulnerabilities.
    -   **`_parse_output(self, output_path: str) -> DFTResult`**: This method reads the output file from a successful DFT run. It will use robust regular expressions or ASE's file parsers to extract the final total energy, the forces on each atom, and the virial stress tensor. It will populate and return a `DFTResult` Pydantic model.
    -   **`_handle_convergence_error(self, log_content: str, current_params: dict) -> dict`**: This is the core of the auto-recovery logic. It takes the log output from a failed run and the parameters that were used. It contains a series of checks for common error messages (e.g., "convergence NOT achieved"). Based on the error type, it returns a *new* dictionary of parameters with modified values (e.g., `mixing_beta` reduced from 0.7 to 0.3). If no known error is found or all retry strategies are exhausted, it will return `None`, signalling a fatal error.
    -   **`_get_heuristic_parameters(self, atoms: ase.Atoms) -> dict`**: This is a crucial helper that encapsulates domain knowledge. It determines calculation parameters based on the input structure. It will contain sub-logic to:
        -   Read from a data file (e.g., a JSON representation of the SSSP library) to determine the recommended plane-wave and charge density cutoffs for the elements present in `atoms`.
        -   Automatically determine if spin-polarised calculations are needed by checking for the presence of magnetic elements (e.g., Fe, Co, Ni).
        -   Calculate an appropriate k-point grid density based on the lattice dimensions, ensuring a consistent sampling of the reciprocal space.
        -   Set smearing parameters, which are essential for calculations involving metals.

**Component Blueprint: `utils/ase_utils.py`**

This utility module provides a simple, focused API for interacting with the project's database.

-   **`save_dft_result(db_path: Path, atoms: ase.Atoms, result: DFTResult)`**: This function connects to the ASE database (e.g., an SQLite file) at the given path. It takes the original `ase.Atoms` object and the corresponding `DFTResult` object. It attaches the energy, forces, and stress from the result to the `atoms.info` dictionary and then uses the `db.write()` method to save the structure and its properties. It will also add important metadata, such as a unique ID for the calculation and a `config_type` label (e.g., 'initial_training_set').
-   **`check_if_exists(db_path: Path, atoms: ase.Atoms) -> bool`**: A helper function to prevent duplicate calculations. Before running a new DFT calculation, the workflow can use this to check if an identical or very similar structure already exists in the database.

This modular design ensures a clear separation of concerns: the `DFTFactory` knows how to run DFT, and `ase_utils` knows how to save the results. This makes them independently testable and easier to maintain.

## 3. Design Architecture

The design of Cycle 1 is fundamentally rooted in a **Schema-First** philosophy, where the data structures that flow between components are rigorously defined before implementation. This is achieved using Pydantic, which enforces type safety and validation at runtime, preventing a wide range of common data-related bugs.

**Pydantic Schema Definitions:**

The following Pydantic models will be defined in a new file, `mlip_autopipec/config/models.py`, to serve as the single source of truth for the data architecture of this cycle.

1.  **`DFTInputParameters(BaseModel)`**: This model represents the complete set of parameters required to define a Quantum Espresso calculation. It ensures that all settings are valid before a calculation is even attempted.
    -   `calculation_type: Literal['scf'] = 'scf'`: A key invariant. The system is designed for single-point calculations only; this prevents accidental structural relaxations.
    -   `pseudopotentials: Dict[str, str]`: A dictionary mapping element symbols (e.g., "Si") to their pseudopotential filenames (e.g., "Si.upf").
    -   `cutoffs: CutoffConfig`: A nested model containing `wavefunction` and `density` cutoffs.
    -   `k_points: Tuple[int, int, int]`: The k-point mesh dimensions.
    -   `smearing: SmearingConfig`: A nested model for smearing type (e.g., 'mv') and width.
    -   `magnetism: Optional[MagnetismConfig] = None`: An optional nested model to define spin-polarization and initial magnetic moments.
    -   `mixing_beta: float = Field(0.7, gt=0.0, le=1.0)`: Example of validation. The mixing parameter must be between 0 and 1.
    -   `model_config = ConfigDict(extra='forbid')`: This is a critical setting that prevents any extra, undefined fields from being passed into the model, catching typos in configuration files.

2.  **`DFTJob(BaseModel)`**: Represents a single, self-contained DFT job to be executed.
    -   `atoms: Any`: The `ase.Atoms` object. Pydantic v2 can handle arbitrary types like this. A custom validator will ensure it is a genuine `Atoms` object.
    -   `params: DFTInputParameters`: The validated input parameters for this specific job.
    -   `job_id: UUID = Field(default_factory=uuid4)`: A unique identifier for tracking and logging.

3.  **`DFTResult(BaseModel)`**: Represents the output of a successful DFT calculation. This is the primary data transfer object returned by the `DFTFactory`.
    -   `job_id: UUID`: Links the result back to the job that produced it.
    -   `energy: float`: The final, converged total energy in eV.
    -   `forces: List[List[float]]`: A nested list representing the forces on each atom. A `@field_validator` will be used to ensure the dimensions are correct (N_atoms x 3).
    -   `stress: List[float]`: A list of the 6 unique components of the virial stress tensor (Voigt notation).

**Data Flow and Consumers:**

-   **Producer:** The `WorkflowManager` (to be implemented in a future cycle) will be the primary producer of `DFTJob` objects. It will construct the `ase.Atoms` object and delegate the creation of the `DFTInputParameters` to the `DFTFactory`'s heuristic engine.
-   **Consumer:** The `DFTFactory` is the consumer of `DFTJob` objects. Its `run` method will accept this object as input.
-   **Producer:** The `DFTFactory` is the producer of `DFTResult` objects.
-   **Consumer:** The `ase_utils.py` module is the primary consumer of `DFTResult` objects, persisting them to the database.

**Invariants and Constraints:**
-   The system will strictly enforce `calculation = 'scf'`. The purpose of the workflow is to learn the potential energy surface as it is given; relaxing the input structures would discard valuable information about high-energy configurations.
-   Element symbols provided in any configuration must be valid chemical symbols (e.g., checked against `ase.data.chemical_symbols`). This prevents typos from causing cryptic errors deep inside the DFT code.
-   The shapes and dimensions of numerical data like forces and stresses will be validated by Pydantic models, ensuring consistency before they are saved to the database.

This schema-driven design ensures that any developer working with these components has a clear and unambiguous contract for the data they need to provide and can expect to receive. It makes the system more robust, easier to debug, and simpler to reason about.

## 4. Implementation Approach

The implementation of Cycle 1 will proceed in a logical, step-by-step manner, building from the data models up to the final execution logic. This approach ensures that each layer is stable before the next is built on top of it.

1.  **Define Pydantic Models:** The first step is to create the `mlip_autopipec/config/models.py` file and implement the `DFTInputParameters`, `DFTJob`, and `DFTResult` models as described in the Design Architecture. This includes adding field types, default values, and validation rules. This provides the data contracts for the rest of the cycle.

2.  **Implement Database Utilities:** Create the `mlip_autopipec/utils/ase_utils.py` file. Implement the `save_dft_result` function. This will initially be a simple wrapper around the `ase.db.connect` and `db.write` methods. The function will take the `DFTResult` model and attach its contents to the `atoms.info` dictionary before writing, ensuring a standardized storage format.

3.  **Create the `DFTFactory` Skeleton:** In `mlip_autopipec/modules/dft.py`, create the `DFTFactory` class. Implement the `__init__` method to accept a configuration object and the public `run` method with the correct signature. Initially, the `run` method can be a placeholder.

4.  **Implement Heuristic Parameter Logic:** Implement the `_get_heuristic_parameters` private method. This will involve:
    -   Creating a small JSON file in a `data` directory containing SSSP pseudopotential and cutoff information.
    -   Writing the Python code to read this JSON and select the appropriate cutoffs based on the elements in an input `ase.Atoms` object.
    -   Implementing the logic to check for magnetic elements and add magnetism parameters if needed.
    -   Implementing the k-point density calculation based on the `atoms.cell.lengths()`.

5.  **Implement Input File Generation:** Implement the `_prepare_input_files` method. This method will call `_get_heuristic_parameters` to get the parameters, then use the `ase.calculators.espresso.Espresso` calculator object to generate the Quantum Espresso input file. ASE handles much of the complexity of formatting the file correctly.

6.  **Implement DFT Execution:** Implement the `_execute_dft` method. This will be a wrapper around `subprocess.run`. The command to be executed will be constructed from the configuration (e.g., `mpirun -np 16 pw.x -in input.pwi > output.pwo`). The function will be configured to block until the subprocess completes and to raise an exception if the return code is non-zero.

7.  **Implement Output Parsing:** Implement the `_parse_output` method. It will use `ase.io.read` with `format='espresso-out'` to parse the output file. This function is robust and can extract energy, forces, and stress. The extracted data will be used to populate and return a `DFTResult` object.

8.  **Implement the `run` Method Orchestration:** Flesh out the main `run` method. It will chain together the calls to the private methods: `_prepare_input_files`, `_execute_dft`, and `_parse_output`. It should be wrapped in a `try...except` block to catch failures from the execution step.

9.  **Implement Auto-Recovery Logic:** Implement the `_handle_convergence_error` method. This method will contain a series of `if/elif` statements that search for specific error strings in the captured `stderr`/`stdout` of a failed run. For example: `if "convergence NOT achieved" in log_content:`. If an error is found, it will return a *new* dictionary of parameters to be used in a retry attempt.

10. **Integrate Retry Loop in `run`:** Modify the `run` method to include the retry loop. It will be a `for` loop that iterates, for example, 3 times. Inside the loop, it will call the execution and parsing methods. If an execution error occurs, it will call `_handle_convergence_error`. If that method returns a new set of parameters, the loop will continue to the next iteration; otherwise, it will break and re-raise the exception.

By following these steps, the `DFTFactory` will be built incrementally, allowing for testing at each stage of development.

## 5. Test Strategy

The testing strategy for Cycle 1 is divided into two distinct but complementary approaches: unit testing to verify individual components in isolation, and integration testing to ensure they work together correctly with external dependencies like Quantum Espresso.

**Unit Testing Approach (Min 300 words):**

The primary goal of unit testing is to validate the internal logic of the `DFTFactory` without ever needing to run a real DFT calculation. This makes the tests extremely fast and allows for precise testing of edge cases and error conditions. The `tests/modules/test_dft.py` file will contain these tests, leveraging `pytest` and mocking libraries.

-   **Testing Heuristics:** The `_get_heuristic_parameters` method will be tested by feeding it various `ase.Atoms` objects and asserting that the returned parameters are correct. For example, a test will be created for a Silicon (Si) crystal, and we will assert that the returned wavefunction cutoff matches the value specified for Si in our mock SSSP JSON data. Another test will use an Iron (Fe) atom and assert that `nspin=2` is correctly added to the parameters. A third test will check that a larger unit cell results in a smaller k-point grid (e.g., `[2, 2, 2]`) compared to a smaller cell (e.g., `[4, 4, 4]`).

-   **Testing Error Recovery:** The `_handle_convergence_error` method is a perfect candidate for unit testing. We will create strings that mimic the `stdout` of a failed Quantum Espresso run. For instance, a test named `test_recovery_on_convergence_failure` will pass a multi-line string containing "convergence NOT achieved" to the method and assert that the returned dictionary contains a reduced `mixing_beta` (e.g., 0.3). Another test will simulate a "Cholesky" error and assert that the `diagonalization` algorithm is changed from `'david'` to `'cg'`. We will also test the case where an unknown error is passed, asserting that the method returns `None`.

-   **Mocking Subprocesses:** To test the main `run` method's orchestration logic, we will use `unittest.mock.patch` to mock `subprocess.run`. This allows us to simulate different outcomes of the DFT execution. One test will mock a successful run, providing fake output data, and we will assert that the `_parse_output` method is called and a `DFTResult` is returned. A separate test will mock a failed run by making `subprocess.run` raise an exception. We will then assert that the `_handle_convergence_error` method is called and that the system attempts to run the subprocess again within the retry loop.

**Integration Testing Approach (Min 300 words):**

While unit tests are essential for logic, integration tests are crucial for verifying that the generated input files are correct and that the interaction with the external Quantum Espresso binary works as expected. These tests will be slower and will be marked as such in `pytest` (e.g., with `@pytest.mark.integration`) so they can be run separately from the fast unit tests.

-   **End-to-End "Happy Path" Test:** The most important integration test is a simple, successful calculation. The test will create an `ase.Atoms` object for a well-behaved, simple system like a 2-atom Silicon unit cell. It will then instantiate a real `DFTFactory` (configured with the path to a locally installed `pw.x`). The test will call the `run` method and let it execute a genuine DFT calculation. The assertions will be on the result: we will check that the returned `energy` is a floating-point number and is within a reasonable tolerance (e.g., ±1%) of a known reference value for that system. We will also assert that the `forces` array has the correct shape `(2, 3)`.

-   **Database Integration Test:** This test will verify the interaction with the database utility. After the "Happy Path" test successfully completes, the test will call `save_dft_result`. It will then use the `ase.db` API to connect to the newly created database file and read the last entry. The test will assert that the `energy` and `forces` stored in the database for that entry match the values returned by the `DFTFactory`. This confirms that the data persistence layer is working correctly.

-   **Real Failure Scenario (Optional but valuable):** If possible, a test could be designed to trigger a real, but recoverable, convergence failure. This could be done by creating a structure with slightly unreasonable bond lengths or setting an aggressive initial `mixing_beta` in the configuration. The test would then assert that the `DFTFactory` successfully recovers and eventually returns a valid result, by checking the logs for evidence of a retry. This provides the ultimate confidence in the auto-recovery mechanism.
