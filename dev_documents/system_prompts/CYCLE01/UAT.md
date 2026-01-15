# MLIP-AutoPipe: Cycle 01 User Acceptance Testing

- **Cycle**: 01
- **Title**: The Foundation - Schemas and DFT Factory Core
- **Status**: Design

---

## 1. Test Scenarios

User Acceptance Testing (UAT) for Cycle 01 is designed to be a developer-centric tutorial. Its goal is to demonstrate the foundational capabilities of the system—configuration management and core DFT execution—and to build confidence in the architectural choices made. The UAT will be presented as a single, narrative-driven Jupyter Notebook (`01_core_functionality.ipynb`). This format is chosen to provide a hands-on, exploratory experience, allowing the "user" (in this case, a fellow developer or system integrator) to execute code snippets, inspect the objects created, and directly verify that the system's core components behave exactly as specified. This notebook serves not only as a test but also as the first piece of developer documentation, showcasing how to programmatically interact with the core objects of the MLIP-AutoPipe system.

---

### **Scenario ID: UAT-C01-01**
- **Title**: Verifying Configuration Schema and DFT Execution
- **Priority**: Critical

**Description:**
This scenario guides the user through the entire workflow of Cycle 01. It demonstrates how the Pydantic schemas enforce correctness, how a configuration object is used to control a DFT calculation, and how the results are reliably parsed and stored. The user will interact with the `SystemConfig`, the `QEProcessRunner`, and the `DatabaseManager` to perform a single, validated, end-to-end calculation for a simple material (a single Nickel atom). This provides tangible proof that the foundational building blocks of the entire MLIP-AutoPipe system are robust, predictable, and working as designed.

**UAT Steps via Jupyter Notebook (`01_core_functionality.ipynb`):**

**Part 1: The Power of Schemas**
*   The notebook will begin by importing the `SystemConfig` and `UserConfig` models.
*   **Step 1.1 (Happy Path):** We will instantiate `UserConfig` with a valid, minimal input for a Nickel system. Then, we will show how this is programmatically expanded into the comprehensive `SystemConfig`, and we will print a few key default parameters (e.g., `config.dft.control.verbosity`, `config.dft.system.ecutwfc`) to demonstrate the automated, heuristic-driven parameterisation.
*   **Step 1.2 (Failure Path):** To showcase the system's robustness, we will then attempt to create a `UserConfig` with an invalid `composition` (fractions don't sum to 1.0). The notebook cell will execute this, and the expected output will be a clean `pydantic.ValidationError`. This amazes the user by proving that the system is self-protecting against simple user errors, catching them early with clear messages.

**Part 2: Executing a DFT Calculation**
*   **Step 2.1 (Setup):** We will create a simple ASE `Atoms` object representing a single Ni atom in a periodic box.
*   **Step 2.2 (Instantiation):** We will instantiate the `QEProcessRunner` with the valid `SystemConfig` created in Step 1.1.
*   **Step 2.3 (Execution):** The notebook will call `runner.run(atoms)`. **Crucially, this will use a mocked backend.** The UAT is not about testing Quantum Espresso, but our system's interaction with it. The notebook will explain that a `pytest-mock` patch is active, simulating a successful QE run and returning pre-canned output. This keeps the UAT fast and self-contained.
*   **Step 2.4 (Verification):** We will print the `results` dictionary from the returned `Atoms` object. The user will see the correctly parsed energy, forces, and stress, verifying the parsing logic.

**Part 3: Data Persistence**
*   **Step 3.1 (Setup):** We will instantiate the `DatabaseManager`, pointing it to a temporary file (`test_db.sqlite`).
*   **Step 3.2 (Writing):** We will call `db_manager.write_calculation(result_atoms, metadata={'config_type': 'uat_c01'})`.
*   **Step 3.3 (Verification):** Finally, we will use the `ase.db.connect` function directly to open the database file. We will retrieve the last entry and print its contents. The user will see not only the standard ASE data but also our custom `key_value_pairs`, including the `config_type`. This provides definitive proof that our data persistence layer is working correctly and extending the ASE DB as designed. The notebook will conclude by cleaning up the created database file.

---

## 2. Behavior Definitions

These behaviors describe the expected outcomes of the UAT scenario in a Gherkin-style format. They precisely define the contract that the Cycle 01 implementation must fulfill.

**Feature: Core System Foundation**
As a developer, I want to ensure that the system's configuration is strictly validated and that the DFT factory can reliably execute and persist a calculation, so that I can build more complex modules on a stable foundation.

---

**Scenario: Validating Configuration Schemas**

*   **GIVEN** I am a developer working within the Jupyter Notebook environment.
*   **AND** I have imported the `UserConfig` and `SystemConfig` Pydantic models.
*   **WHEN** I instantiate `UserConfig` with a valid dictionary, for example `{'elements': ['Ni'], 'composition': {'Ni': 1.0}}`.
*   **THEN** a `UserConfig` object is successfully created without errors.
*   **AND** I can then create a `SystemConfig` from this `UserConfig`.
*   **AND** the `SystemConfig` object contains the expected default DFT parameters, such as `config.dft.system.nspin` being `2` because Nickel is a magnetic element.

---

**Scenario: Rejecting Invalid Configuration**

*   **GIVEN** I am a developer working within the Jupyter Notebook environment.
*   **AND** I have imported the `UserConfig` Pydantic model.
*   **WHEN** I attempt to instantiate `UserConfig` with an invalid `composition` where the fractions do not sum to 1.0 (e.g., `{'Ni': 0.8, 'Fe': 0.3}`).
*   **THEN** the system MUST raise a `pydantic.ValidationError`.
*   **AND** the error message MUST clearly indicate that the composition fractions do not sum to 1.0.

---

**Scenario: Successful Mocked DFT Execution and Parsing**

*   **GIVEN** I have a valid `SystemConfig` object for a Nickel system.
*   **AND** I have created an ASE `Atoms` object for a single Ni atom.
*   **AND** the `subprocess.run` call inside the `QEProcessRunner` is mocked to simulate a successful Quantum Espresso execution.
*   **AND** the mock is configured to return a standard QE output string containing known values for energy, forces, and stress.
*   **WHEN** I instantiate `QEProcessRunner` with the `SystemConfig` and call its `run` method with the `Atoms` object.
*   **THEN** the `run` method returns the `Atoms` object.
*   **AND** the returned object's `.calc.results` dictionary contains the keys `'energy'`, `'forces'`, and `'stress'`.
*   **AND** the values for these keys match the known values from the mocked output string, with units correctly converted to eV and Ångströms.

---

**Scenario: Persisting Calculation Results with Custom Metadata**

*   **GIVEN** I have an ASE `Atoms` object that contains the results from a successful (mocked) DFT calculation.
*   **AND** I have instantiated a `DatabaseManager` pointing to a new, temporary SQLite database file.
*   **WHEN** I call the `db_manager.write_calculation` method, passing the `Atoms` object and a metadata dictionary `{'config_type': 'uat_c01', 'custom_id': 'abc'}`.
*   **THEN** the method completes without errors.
*   **AND** when I connect to the SQLite file and retrieve the last row written.
*   **AND** that row's `key_value_pairs` correctly contains the key `'mlip_config_type'` with the value `'uat_c01'`.
*   **AND** it also contains the key `'mlip_custom_id'` with the value `'abc'`.
