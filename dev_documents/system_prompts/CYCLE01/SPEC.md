# MLIP-AutoPipe: Cycle 01 Specification

- **Cycle**: 01
- **Title**: The Foundation - Schemas and DFT Factory Core
- **Status**: Scoping

---

## 1. Summary

This first development cycle is the most critical as it lays the architectural foundation for the entire MLIP-AutoPipe system. The primary objective of Cycle 01 is to establish a robust, schema-driven framework for managing all system configurations and to implement the core component responsible for interacting with the DFT engine. This cycle is not about generating complex scientific workflows but about building the boring, reliable, and testable bedrock upon which all future modules will depend.

The centrepiece of this cycle is the implementation of a comprehensive suite of Pydantic models. We will follow a strict **Schema-First Development** process. Before any logic is written, we will meticulously define the data structures that govern the system. This includes the `UserConfig`, which represents the minimal, user-friendly input, and the `SystemConfig`, the massive, fully-parameterised internal configuration object. A heuristic engine will be designed to translate the former into the latter, encapsulating expert knowledge and sensible defaults, thereby abstracting complexity away from the end-user. This approach ensures that all data flowing through the system is validated, typed, and explicit, which dramatically reduces the potential for runtime errors.

The second major deliverable is the initial implementation of the `QEProcessRunner` class. This component is the system's sole gateway to the Quantum Espresso DFT code. This initial version will focus on the "happy path": taking a single atomic structure (an ASE `Atoms` object) and a `SystemConfig`, generating the correct QE input file, executing the `pw.x` binary, and parsing the output to extract the essential physical quantities—total energy, atomic forces, and the virial stress tensor. While the full multi-level error recovery is deferred to a later cycle, the initial implementation will include basic error detection.

Finally, this cycle will establish the data persistence layer. A `DatabaseManager` will be created to interface with an ASE-compatible database (SQLite for simplicity in this cycle). Crucially, this manager will extend the standard ASE DB schema to include the custom metadata fields required by our active learning loop, such as the configuration's origin (`config_type`) and the `force_mask`. By the end of this cycle, we will have a demonstrable, albeit simple, workflow: a developer can define a calculation via a Pydantic model, execute a single-point DFT calculation, and see the results correctly persisted in the custom-schema database. This provides the fundamental building blocks for all subsequent development.

---

## 2. System Architecture

The work in Cycle 01 focuses on creating the initial files and structures for the core configuration, the DFT factory module, and the data persistence layer. This establishes the central spine of the application.

**File Structure for Cycle 01:**

The following files will be created or modified in this cycle. New files are marked in **bold**.

```
.
├── pyproject.toml
└── src/
    └── mlip_autopipec/
        ├── __init__.py
        ├── config/
        │   ├── __init__.py
        │   ├── **user.py**         # Schema for the minimal user input (input.yaml)
        │   └── **system.py**       # Schema for the fully-expanded system configuration
        ├── modules/
        │   ├── __init__.py
        │   └── **dft_factory.py**  # Module C: QEProcessRunner class
        ├── data/
        │   ├── __init__.py
        │   └── **database.py**     # Wrapper for ASE DB with custom metadata handling
        └── utils/
            └── __init__.py
```

**Component Breakdown:**

*   **`pyproject.toml`**: This file will be populated with the initial set of project dependencies required for this cycle, including `pydantic`, `ase`, `numpy`, and `typer`. Development dependencies like `pytest` and `pytest-mock` will also be added.

*   **`config/user.py`**: This file will contain the Pydantic model for the `UserConfig`. It defines the public-facing API of the system, focusing on simplicity and intent. Fields will be high-level, such as `elements`, `composition`, and `simulation_goal`.

*   **`config/system.py`**: This file will house the extensive `SystemConfig` model and its various sub-models (e.g., `DFTParams`, `MDParams`). This is the internal, "source-of-truth" configuration object. It will contain dozens of fields, each with a sensible default value, covering everything from QE's `mixing_beta` to the convergence thresholds for electronic steps.

*   **`modules/dft_factory.py`**: This file will contain the `QEProcessRunner` class. It will be initialized with a `SystemConfig` object. Its primary public method, `run()`, will accept an ASE `Atoms` object. Internally, it will have private methods for `_generate_input_file()`, `_execute_pw_x()`, and `_parse_output()`. This clear separation of concerns within the class will make it highly testable. The class will handle the conversion of ASE's units (eV, Å) to Quantum Espresso's units (Ry, Bohr).

*   **`data/database.py`**: This file will define the `DatabaseManager` class. It will contain methods like `connect()`, `write_calculation()`, and `get_completed_calculations()`. The `write_calculation()` method will be the most critical; it will take an ASE `Atoms` object (with calculation results attached) and a dictionary of custom metadata (`config_type`, `uncertainty_gamma`, etc.) and write everything to the underlying database in a single transaction. This ensures data integrity and atomicity.

This architecture ensures a clean separation of concerns: configuration is handled by the `config` package, execution logic by the `modules` package, and data persistence by the `data` package.

---

## 3. Design Architecture

The design of Cycle 01 is anchored in the **Schema-First Development** philosophy, where the data models are the primary design artefact. All logic is then built to be a consumer or producer of these strictly-defined models.

**Pydantic Schema Design (`user.py` and `system.py`):**

*   **`UserConfig`**: This model is designed for the user. It uses expressive types and validators to provide clear error messages. For example:
    *   `composition`: A dictionary mapping element symbols to their fractional values. A Pydantic validator will ensure the fractions sum to 1.0.
    *   `simulation_goal`: An `Enum` type that restricts the user's choice to a list of supported goals like `'melt_quench'` or `'elastic'`.
    *   **Invariants**: The set of elements derived from the `composition` dictionary must be consistent with the `elements` list.

*   **`SystemConfig`**: This is the exhaustive internal model. It is designed for the developer.
    *   **Nested Structure**: The configuration will be broken down into logical, nested Pydantic models. For example, the main `SystemConfig` will have an attribute `dft: DFTParams`, where `DFTParams` is another `BaseModel` containing all QE-specific settings. This makes the configuration navigable and prevents it from becoming a monolithic, flat namespace.
    *   **Strict Validation**: Every model will use `model_config = ConfigDict(extra="forbid")`. This is a critical design choice that prevents typos or incorrect parameters from being silently ignored. If a parameter is not explicitly defined in the schema, the system will raise a validation error, forcing developers to be explicit and correct.
    *   **Producers and Consumers**: The `HeuristicEngine` (to be implemented in a future cycle) will be the sole **producer** of the `SystemConfig`. All other modules, starting with `QEProcessRunner`, will be strict **consumers**. They receive the `SystemConfig` at initialization and treat it as a read-only object. This unidirectional data flow is key to making the system's behaviour predictable.
    *   **Extensibility**: The nested design allows for easy extension. For instance, adding a new DFT code would involve creating a new `VaspParams` model and adding it to a `Union` type within the `SystemConfig`, without breaking the existing `QEProcessRunner` which only consumes `DFTParams`.

**`QEProcessRunner` Design (`dft_factory.py`):**

*   **Responsibility**: This class's single responsibility is to execute a single-point DFT calculation. It knows nothing about active learning, databases, or generators.
*   **Interface**: The public API will be minimal: `__init__(self, config: SystemConfig)` and `run(self, atoms: Atoms) -> Atoms`. The `run` method returns the input `Atoms` object with the calculation results (`energy`, `forces`, `stress`) attached to its `.calc.results` dictionary. This is the standard ASE convention and ensures interoperability.
*   **State Management**: The `QEProcessRunner` will be stateless. Each call to `run()` is an independent, atomic operation. All necessary information is provided through the `config` and `atoms` arguments. This makes the class thread-safe and suitable for parallel execution in later cycles.

**`DatabaseManager` Design (`database.py`):**

*   **Wrapper, Not a Framework**: This class is a thin, convenient wrapper around the `ase.db` API. It is not an ORM.
*   **Custom Metadata Handling**: The key design feature is its explicit handling of the custom metadata columns. The `write_calculation` method signature will be `write_calculation(self, atoms: Atoms, metadata: dict)`. It will use the standard `ase.db.connect.write()` method but will also explicitly add the key-value pairs from the `metadata` dictionary into the `key_value_pairs` field of the database row. This keeps our custom data neatly namespaced and prevents conflicts with standard ASE columns.

This design ensures that each component in Cycle 01 is small, focused, and has a well-defined interface, making the system easy to test, debug, and build upon.

---

## 4. Implementation Approach

The implementation will proceed in a logical, bottom-up fashion, starting with the data schemas and progressively building up to the execution logic.

1.  **Project Setup:**
    *   Create the directory structure as outlined in the System Architecture section using `mkdir -p`.
    *   Initialise `pyproject.toml` and add initial dependencies: `pydantic`, `ase`, `numpy`, `typer`.
    *   Add development dependencies: `pytest`, `pytest-mock`, `ruff`, `mypy`.
    *   Create the `uv` virtual environment and install all dependencies.

2.  **Pydantic Schema Implementation (`user.py`, `system.py`):**
    *   Begin with `user.py`. Define the `TargetSystem` and `Resources` sub-models first, then compose them into the main `UserConfig` model. Add validators for fields like `composition`.
    *   Move to `system.py`. Define the lowest-level nested models first (e.g., `DFTControlParams`, `DFTElectronsParams`).
    *   Compose these into the main `DFTParams` model.
    *   Finally, create the top-level `SystemConfig` model that includes `DFTParams` and placeholders for other modules (`GeneratorParams`, `TrainerParams`, etc.).
    *   Ensure every model has the strict `ConfigDict(extra="forbid")`.

3.  **Database Manager Implementation (`database.py`):**
    *   Create the `DatabaseManager` class.
    *   The `__init__` method will take the database path from the `SystemConfig`.
    *   The `connect` method will establish the connection using `ase.db.connect()`.
    *   Implement the `write_calculation` method. It will first prepare the custom metadata by prefixing keys (e.g., `mlip_config_type`) to avoid clashes. It will then call the underlying ASE `write()` function, passing the `Atoms` object and the prepared dictionary to the `key_value_pairs` argument.

4.  **Core DFT Runner Implementation (`dft_factory.py`):**
    *   Create the `QEProcessRunner` class, which takes a `SystemConfig` in its constructor.
    *   Implement the `_generate_input_file` private method. This is the most complex part of this cycle. It will programmatically build the QE input string by reading parameters from the `self.config.dft` object. It will involve creating formatted strings for sections like `&CONTROL`, `&SYSTEM`, `ATOMIC_SPECIES`, `ATOMIC_POSITIONS`, and `K_POINTS`.
    *   Implement the `_execute_pw_x` method. This method will use Python's `subprocess.run` to execute the `pw.x` command. It will capture `stdout` and `stderr` and check the return code for basic success or failure.
    *   Implement the `_parse_output` method. This method will parse the `stdout` string from the QE run. It will use regular expressions to find the final total energy, the atomic forces, and the stress tensor. It must handle unit conversions from Ry to eV and Bohr to Ångström.
    *   Implement the public `run` method. This method will orchestrate the calls to the private methods in the correct order. It will attach the parsed results to the returned `Atoms` object.

5.  **Initial Testing:**
    *   Write unit tests for the Pydantic schemas in `tests/config/test_schemas.py`. Test that default values are set correctly and that validation errors are raised for incorrect inputs.
    *   Write unit tests for `QEProcessRunner` in `tests/modules/test_dft_factory.py`. The focus will be on testing the `_generate_input_file` method. Create a fixture with a sample `SystemConfig` and `Atoms` object and assert that the generated input string is exactly correct.
    *   The execution and parsing methods will be tested using integration tests, which will be the primary focus of the test strategy for this cycle.

This step-by-step process ensures that each piece of functionality is built upon a solid, already-tested foundation.

---

## 5. Test Strategy

Testing in Cycle 01 is paramount to ensure the stability of the entire project. The strategy is divided into rigorous unit testing of the data structures and isolated logic, and integration testing of the component that interacts with the external world (the DFT process).

**Unit Testing Approach (Min 300 words):**

The primary focus of unit testing in this cycle is on the components with complex internal logic that do not depend on external processes: the Pydantic schemas and the QE input file generation.

*   **Pydantic Schema Testing (`tests/config/test_schemas.py`):**
    We will create a dedicated test file for our configuration models. The tests will verify several key aspects of the schemas. Firstly, we will test **default value instantiation**. We will create an instance of `SystemConfig` from an empty `UserConfig` (mimicking the eventual Heuristic Engine's input) and assert that all the deeply nested default values (e.g., `config.dft.control.verbosity`) are set to their expected states. This ensures our baseline calculations are predictable. Secondly, we will exhaustively test the **validation logic**. For `UserConfig`, we'll test that a `composition` dictionary whose values do not sum to 1.0 raises a `ValidationError`. We will test that providing an unsupported `simulation_goal` string also fails validation. For `SystemConfig`, we will test the `extra="forbid"` configuration by attempting to initialize it with a dictionary containing a misspelled or non-existent parameter, and we will assert that a `ValidationError` is raised. This is crucial for preventing silent configuration errors.

*   **DFT Input Generation Testing (`tests/modules/test_dft_factory.py`):**
    The logic for generating the Quantum Espresso input file is complex and prone to formatting errors. Therefore, it must be unit-tested in complete isolation. We will create a `pytest` fixture that provides a standardised `SystemConfig` object and a simple `Atoms` object (e.g., a single Nickel atom in a cubic box). The test function will instantiate `QEProcessRunner` with this config and call the (normally private) `_generate_input_file` method. The output string will be compared against a "golden" input file stored as a multi-line string in the test file. The comparison will be exact, character-for-character. This test will ensure that any refactoring of the generation logic doesn't accidentally change the output format, which is critical for reproducibility. We will have separate tests for different configurations, for example, one for a metallic, magnetic system (which should include `nspin=2` and `smearing='mv'`) and one for a simple insulator.

**Integration Testing Approach (Min 300 words):**

The integration tests will verify the `QEProcessRunner`'s ability to correctly manage an external subprocess and parse its output. These tests will rely heavily on mocking to avoid actually running a time-consuming DFT calculation.

*   **Mocked Process Execution (`tests/modules/test_dft_factory.py`):**
    The core integration test will use the `mocker` fixture from `pytest-mock` to patch Python's `subprocess.run`. We will set up the mock to return a `CompletedProcess` object with a specific `stdout`, `stderr`, and `returncode`.
    1.  **"Happy Path" Test:** The first test will simulate a successful QE run. The mock will be configured with `returncode=0` and a `stdout` attribute containing a sample of a real, successful QE output. We will then call the public `runner.run(atoms)` method. The assertions will be threefold: First, we will assert that `subprocess.run` was called with the expected command-line arguments (e.g., `['pw.x', '-in', 'some_file.in']`). Second, we will assert that the `_parse_output` method correctly extracted the energy, forces, and stress from the sample `stdout`. Third, we will assert that the returned `Atoms` object has these results correctly stored in its `.calc.results` dictionary.
    2.  **Failure Path Test:** Another test will simulate a failed QE run. We will configure the mock to return a non-zero `returncode`. In this scenario, we will assert that the `runner.run()` method raises a custom exception (e.g., `DFTCalculationError`). This verifies our basic error-handling mechanism.

*   **Database Interaction Test:**
    We will also test the integration between the `QEProcessRunner` and the `DatabaseManager`. This test will use a temporary SQLite database file. It will execute the "happy path" test described above, but after the `runner.run()` call, it will instantiate a `DatabaseManager` and use it to write the results. Finally, it will connect to the temporary database directly and read the last row, asserting that the energy, forces, and our custom metadata (e.g., `{'config_type': 'test'}`) have been persisted correctly. This test validates the entire data flow for a single, successful calculation.
