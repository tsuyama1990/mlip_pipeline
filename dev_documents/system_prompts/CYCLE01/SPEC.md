# CYCLE01 Specification: The Foundation

## 1. Summary

This document provides the detailed technical specification for CYCLE01 of the MLIP-AutoPipe project. The focus of this cycle is to construct the foundational bedrock of the entire system. This involves two critical components: **Module C, the Automated DFT Factory**, and **Module A, the Physics-Informed Generator**. The primary objective of this cycle is to establish a robust, automated pipeline capable of generating high-quality, physically diverse atomic structures and then calculating their ground-truth properties (energy, forces, stress) using Density Functional Theory (DFT) without any human intervention.

**Module C (Automated DFT Factory)** will be a highly reliable wrapper around the Quantum Espresso software package. Its core responsibility is to accept an atomic structure and return its calculated properties. The "zero-human" philosophy mandates that this module must be intelligent enough to handle the complexities of DFT calculations autonomously. This includes selecting appropriate computational parameters (pseudopotentials, cutoffs, k-points, smearing) based on the input structure's elemental composition and geometry, and, crucially, implementing a sophisticated error-recovery mechanism to handle common DFT convergence failures. The module will be designed for high-throughput, parallel execution on HPC clusters, treating DFT calculations as a scalable, automated service.

**Module A (Physics-Informed Generator)** is responsible for creating the initial seed data for the entire active learning process. Relying solely on random structure generation or computationally expensive AIMD is inefficient. Therefore, this module will employ physics-informed strategies to generate a diverse set of structures that effectively sample the potential energy surface. For metallic alloys, it will use the Special Quasirandom Structures (SQS) method combined with lattice strain and random atomic displacements (rattling) to capture elastic and vibrational properties. For molecular systems, it will use Normal Mode Sampling (NMS) to explore the vibrational landscape efficiently. For crystalline systems, it will employ a surrogate-driven Melt-Quench procedure and introduce various point defects to learn about bond breaking, formation, and defect energetics.

By the end of CYCLE01, we will have a command-line-drivable system capable of taking a high-level description of a material and producing a high-quality, diverse, and clean dataset of DFT-calculated structures. This dataset will form the essential starting point for training the first machine learning potential in a subsequent cycle. This cycle lays the non-negotiable groundwork for the intelligence and autonomy that will be built on top of it.

## 2. System Architecture

The system architecture for CYCLE01 is focused on creating the initial data generation pipeline. It establishes the core file structure, data schemas, and module interactions that will be expanded upon in later cycles. The design is modular, with a clear separation between the data definition (schemas), the core logic (modules), and the command-line interface.

**File Structure for CYCLE01:**

The files and directories to be created or modified in this cycle are marked in bold.

```
mlip_autopipec/
├── src/
│   ├── mlip_autopipec/
│   │   ├── __init__.py
│   │   ├── **main.py**             # CLI entry point for testing generation and DFT runs
│   │   ├── **settings.py**         # Global settings, including paths to executables
│   │   ├── **schemas/**
│   │   │   ├── __init__.py
│   │   │   ├── **user_config.py**    # Pydantic models for a simplified input
│   │   │   ├── **system_config.py**  # Pydantic models for the full, expanded config
│   │   │   └── **dft.py**            # Schemas for DFT calculation inputs and outputs
│   │   ├── **modules/**
│   │   │   ├── __init__.py
│   │   │   ├── **a_generator.py**    # Module A: Physics-Informed Generator
│   │   │   └── **c_dft_factory.py**  # Module C: Automated DFT Factory
│   │   └── **utils/**
│   │       ├── __init__.py
│   │       ├── **ase_utils.py**      # Utilities for ASE Atoms objects
│   │       └── **qe_utils.py**       # Utilities for Quantum Espresso
├── **tests/**
│   ├── __init__.py
│   ├── **test_schemas.py**
│   └── **test_modules/**
│       ├── __init__.py
│       ├── **test_a_generator.py**
│       └── **test_c_dft_factory.py**
├── pyproject.toml
└── README.md
```

**Architectural Blueprint:**

The workflow within CYCLE01 will be orchestrated by the `main.py` script, which will serve as a temporary driver for testing and validation purposes. The process is as follows:

1.  A user (or a test script) creates a simplified input YAML file based on the schema in `user_config.py`. This might specify, for example, an `FeNi` alloy.
2.  The `main.py` script reads this YAML and uses a heuristic engine (to be prototyped in `settings.py` or `main.py`) to expand it into a full `SystemConfig` object, defined in `system_config.py`. This object contains all the detailed parameters for the requested job.
3.  The `SystemConfig` object is passed to **Module A (`a_generator.py`)**. This module, based on the configuration, generates a list of `ase.Atoms` objects. For an `FeNi` alloy, this would involve calling `icet` to generate SQS, then applying a series of strains and rattles to each SQS structure. Each generated structure is tagged with metadata (e.g., `config_type='sqs_strained'`).
4.  The list of `ase.Atoms` objects is then passed to **Module C (`c_dft_factory.py`)**.
5.  For each `Atoms` object, the DFT Factory constructs a `DFTInput` object (defined in `dft.py`). This is where the heuristic logic resides. The `qe_utils.py` sub-module inspects the `Atoms` object to determine the elements present, the cell size, and whether it's likely a metal. It then uses this information to select the appropriate SSSP pseudopotentials, calculate a suitable plane-wave cutoff energy, generate a k-point grid, and set the smearing parameters.
6.  The DFT Factory then uses these parameters to run a Quantum Espresso `pw.x` calculation. It manages the execution of the external process, monitors its progress, and parses the output.
7.  If the calculation fails to converge, the **auto-recovery logic** is triggered. The Factory will catch the error, modify the `DFTInput` parameters according to a predefined strategy (e.g., reduce `mixing_beta`), and resubmit the job, up to a maximum number of retries.
8.  Upon successful completion, the Factory parses the final energy, forces, and stress tensor, packaging them into a `DFTOutput` object.
9.  This `DFTOutput` object, along with the original structure and its metadata, is saved to a central data store (initially, this will be an ASE database file, e.g., `structures.db`).

This architecture ensures a clean separation of concerns. The schemas provide a stable, validated interface between modules. The generator module knows nothing about DFT, and the DFT module knows nothing about structure generation. This modularity is essential for building a complex, maintainable system.

## 3. Design Architecture

The design of CYCLE01 is heavily reliant on a Pydantic-based schema to ensure data integrity and to provide a clear, self-documenting structure for the data that flows through the system. This approach is critical for creating a robust and maintainable codebase, especially in a system designed for full autonomy.

**Pydantic Schema Design:**

*   **`user_config.py`**: This defines the public-facing configuration. It is designed to be minimal and intuitive.
    *   `TargetSystem(BaseModel)`: Contains fields like `elements: List[str]`, `composition: Dict[str, float]`, and `crystal_structure: str`. This captures the "what" of the simulation.
    *   `GenerationConfig(BaseModel)`: Specifies the type of generation to perform, e.g., `'alloy_sqs'`, `'molecule_nms'`.
    *   `UserConfig(BaseModel)`: The top-level model that combines the above and includes a `project_name`.
    *   **Invariants**: The sum of composition fractions must equal 1.0. The elements list must match the keys of the composition dict. These will be enforced with Pydantic validators.
    *   **Consumers**: This is consumed by the heuristic engine in `main.py`.

*   **`system_config.py`**: This is the internal, fully-specified configuration. It's the "single source of truth" for a run.
    *   `DFTParams(BaseModel)`: Explicitly defines every key DFT parameter: `pseudopotentials: Dict[str, str]`, `cutoff_wfc: float`, `k_points: Tuple[int, int, int]`, `smearing_type: str`, `degauss: float`, `nspin: int`, etc.
    *   `GeneratorParams(BaseModel)`: Detailed parameters for the generator, e.g., `sqs_supercell_size: List[int]`, `strain_magnitudes: List[float]`, `rattle_std_dev: float`.
    *   `SystemConfig(BaseModel)`: Combines all detailed parameter sets. It's a much larger and more complex model than `UserConfig`.
    *   **Producers**: The heuristic engine produces this model.
    *   **Consumers**: Module A and Module C consume this model to configure their behaviour.

*   **`dft.py`**: Defines the data structures for interacting with the DFT engine.
    *   `DFTInput(BaseModel)`: Represents a single, ready-to-run DFT calculation. It includes the `ase.Atoms` structure (using `Arbitrary=True`) and a `DFTParams` object.
    *   `DFTOutput(BaseModel)`: Represents the results of a successful DFT calculation. It contains `total_energy: float`, `forces: List[List[float]]`, and `stress: List[List[float]]`.
    *   **Versioning**: These models are internal but critical. Any change to the DFT code wrapper or the expected outputs would require a version bump or careful handling, though this is less of a concern in the early cycles. The strict schema ensures that any such change is immediately detected.

This schema-driven design forces a clear definition of the data contracts between different parts of the code. It prevents common bugs like typos in dictionary keys or incorrect data types, which is essential for a system that must run reliably for long periods without human oversight.

## 4. Implementation Approach

The implementation of CYCLE01 will be a step-by-step process, starting with the core utilities and schemas, then building the main modules, and finally connecting them with a temporary command-line driver.

**Step 1: Scaffolding and Schema Definition**
*   Create the file and directory structure as outlined in the System Architecture section.
*   Implement the initial Pydantic models in `schemas/user_config.py`, `schemas/system_config.py`, and `schemas/dft.py`. This is the first step, as it defines the "language" the rest of the system will speak.
*   Write unit tests in `tests/test_schemas.py` to validate the schemas, including the custom validators (e.g., for composition).

**Step 2: Developing `c_dft_factory.py` and its Utilities**
*   Begin with `utils/qe_utils.py`. Write functions that take an `ase.Atoms` object and return DFT parameters. For example, a `get_sssp_recommendations(atoms)` function that looks up the SSSP library data (which will be stored as a JSON or YAML file within the package) and returns the recommended pseudopotentials and cutoffs. Another function, `get_kpoints(atoms)`, will implement the linear density heuristic.
*   In `modules/c_dft_factory.py`, create a `QERunner` class.
*   The `QERunner.run(dft_input: DFTInput)` method will be the main entry point. It will use the `ase.calculators.espresso.Espresso` calculator as a starting point, but will wrap it with more robust logic.
*   The method will first write the QE input file, then execute `pw.x` using Python's `subprocess` module, capturing `stdout` and `stderr`.
*   Implement the error detection logic by parsing the output of `pw.x`. Look for specific error messages like "convergence NOT achieved".
*   Implement the auto-recovery `try...except` block. If a convergence error is detected, a helper method will be called to modify the `dft_input` object (e.g., `new_input = self._recover(dft_input, error_type)`). The `run` method will then recursively call itself with the new input, up to a retry limit.
*   Upon success, implement the parsing logic to extract energy, forces, and stress and return a `DFTOutput` object.
*   Write unit tests in `tests/test_modules/test_c_dft_factory.py` using mock `subprocess` calls to simulate successful runs, failed runs, and recovery scenarios.

**Step 3: Developing `a_generator.py`**
*   In `modules/a_generator.py`, create a main function `generate_structures(config: SystemConfig)`.
*   This function will have a dispatcher that calls the appropriate generation method based on `config.generation_type`.
*   Implement the `_generate_alloy_sqs` method. This will involve using the `icet` library to generate the SQS supercells. It will then loop through the specified strain values and rattle standard deviations, applying these transformations using `ase` and `pymatgen`'s built-in functions.
*   Each generated structure will be an `ase.Atoms` object, and its `info` dictionary will be populated with metadata, e.g., `atoms.info['config_type'] = 'sqs_strain_0.01_rattle_0.05'`.
*   Write unit tests in `tests/test_modules/test_a_generator.py` to ensure that the correct number and type of structures are generated for a given input configuration.

**Step 4: Creating the CLI Driver**
*   In `main.py`, use a library like `Typer` to create a simple CLI.
*   The main command will take the path to a `user_config.yaml` file as input.
*   It will first load and validate the user config using the Pydantic model.
*   It will then implement the heuristic engine to convert the `UserConfig` into a `SystemConfig` object.
*   It will call `a_generator.generate_structures`.
*   It will then loop through the generated structures, instantiate the `QERunner`, and call its `run` method for each structure.
*   The results will be saved to an `ase.db` file. This provides a tangible output for the cycle.

This step-by-step approach ensures that each component is built and tested independently before being integrated, reducing complexity and making debugging easier.

## 5. Test Strategy

The test strategy for CYCLE01 is focused on ensuring the correctness and robustness of the foundational data generation pipeline. Given that this cycle produces the "ground truth" data for all subsequent ML, its reliability is paramount. The strategy combines rigorous unit testing of individual components and integration testing of the complete workflow.

**Unit Testing Approach (Min 300 words):**

The unit testing for CYCLE01 will be fine-grained and will focus on verifying the logic of each component in isolation. This is crucial for building a reliable system, as it allows us to pinpoint failures at the lowest level.

*   **Schemas (`test_schemas.py`):** The Pydantic schemas are the first line of defense against invalid data. Unit tests will be written to ensure they behave as expected. This includes testing the validators: for example, providing a `UserConfig` where the composition fractions do not sum to 1.0 and asserting that a `ValidationError` is raised. We will also test successful validation to ensure that valid data passes through correctly.

*   **Generator (`test_a_generator.py`):** The generator's logic will be tested to confirm that it produces the correct set of structures. For the SQS alloy generator, we will provide a fixed input configuration and a fixed random seed. We will then assert that the number of generated structures is correct (e.g., `num_sqs * num_strains * num_rattles`). We will also perform sanity checks on the generated structures themselves: for a strain test, we will check that the cell volume has indeed changed as expected. For a rattle test, we will confirm that the atomic positions are no longer on their perfect lattice sites. We will mock the calls to external libraries like `icet` where necessary to isolate the logic of our own code.

*   **DFT Factory (`test_c_dft_factory.py`):** This is the most critical set of unit tests in this cycle. We will create a mock `subprocess.run` function to simulate the behaviour of `pw.x`. This allows us to test the `QERunner` without actually running any expensive DFT calculations. We will create test cases for:
    *   **Successful run:** The mock `pw.x` returns a "successful" output file content, and we will assert that the `QERunner` correctly parses the energy, forces, and stress.
    *   **Convergence failure:** The mock `pw.x` returns an error message. We will assert that the `QERunner` detects this error and calls its recovery logic. We can check that the recovery logic correctly modifies the input parameters (e.g., that `mixing_beta` is reduced).
    *   **Retry limit:** We will simulate a scenario where the calculation fails repeatedly and assert that the `QERunner` gives up after the specified number of retries and raises an exception.
    *   **Heuristic parameter selection:** We will test the utility functions in `qe_utils.py` directly. For example, we will create a sample `ase.Atoms` object for `fcc` Nickel and assert that the `get_sssp_recommendations` function returns the correct pseudopotential and that `get_kpoints` returns a reasonable k-point mesh.

**Integration Testing Approach (Min 300 words):**

While unit tests verify the components in isolation, integration tests ensure that they work together correctly. For CYCLE01, the primary integration test will be to run the entire data generation pipeline, from user input to the final database, on a small, controlled, and physically meaningful system.

*   **Test Scenario: Diatomic Molecule (e.g., N2)**
    *   This is an ideal first integration test. It's simple, computationally cheap to run with DFT, and the physics is well-understood.
    *   **Setup:** We will create a `user_config.yaml` that specifies Nitrogen (`N`) and a generation goal of creating a "bond-stretching" curve. The `a_generator` module will be extended with a simple generator that creates a series of `N2` molecules with varying interatomic distances (e.g., from 0.8 Å to 3.0 Å).
    *   **Execution:** We will run the main CLI driver (`main.py`) with this configuration file. This will trigger the entire workflow:
        1.  The generator will create the set of `N2` molecules.
        2.  The DFT factory will be called for each molecule. Since these are real DFT calculations, this test will be marked as a "slow" test and run less frequently than the unit tests (e.g., as part of a nightly build).
        3.  The results will be saved to a temporary `test_n2.db` file.
    *   **Validation:** After the run completes, a validation script will be executed. This script will:
        1.  Read the `test_n2.db` file.
        2.  Assert that the database contains the expected number of structures.
        3.  Extract the interatomic distance and the calculated total energy for each structure.
        4.  Plot the energy vs. distance curve.
        5.  Assert that the curve has the expected shape: a single minimum at the equilibrium bond length (around 1.1 Å) and high energies at very short (repulsive) and very long (dissociated) distances.
        6.  Assert that the calculated energy at the minimum is close to the known experimental or high-precision computational value.

This integration test provides a powerful end-to-end validation of the entire CYCLE01 pipeline. It confirms that the schemas are correct, the generator and DFT factory are working, they are communicating correctly, and that the final output is physically meaningful. This gives us high confidence that the foundation of our system is solid.
