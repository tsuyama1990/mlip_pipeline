# CYCLE02 Specification: The Apprentice

## 1. Summary

This document provides the detailed technical specification for CYCLE02 of the MLIP-AutoPipe project. Building upon the foundational data generation capabilities established in CYCLE01, this cycle introduces the first layer of intelligence and efficiency into the pipeline. The focus is on two new components: **Module B, the Surrogate Explorer**, and the initial version of **Module D, the Pacemaker Trainer**. The main objective of this cycle is to significantly reduce the number of expensive DFT calculations required to build a potential by using a pre-trained "surrogate" model to perform a broad, inexpensive initial search of the configuration space.

**Module B (Surrogate Explorer)** will act as a smart filter between the initial structure generator (Module A) and the DFT factory (Module C). Instead of sending every generated structure for a full DFT calculation, this module will first analyze them using a pre-trained, general-purpose machine learning potential, specifically MACE-MP. This "direct sampling" approach allows the system to quickly discard physically unrealistic structures and to get a rough estimate of the energy landscape at a tiny fraction of the cost of a DFT calculation. Following this initial screening, the module will employ a Farthest Point Sampling (FPS) algorithm to select a structurally diverse and maximally informative subset of candidates from the vast pool of surrogate-assessed structures. This ensures that our limited DFT budget is spent only on the most valuable calculations.

**Module D (Pacemaker Trainer)**, in this cycle, will be a manually-triggered component responsible for taking the curated data from the ASE database and training a new machine learning interatomic potential using the Pacemaker framework. This involves automatically generating the correct input files for Pacemaker, managing the training process, and storing the resulting potential (`.yace` file) in a versioned manner. The training will incorporate a "Delta Learning" strategy, which improves the model's accuracy by training it to predict the difference between the DFT energy and a baseline potential (ZBL), particularly for repulsive short-range interactions.

By the end of CYCLE02, we will have a semi-automated workflow. The system will be able to intelligently select a small, high-value subset of structures for DFT calculation, and a user will then be able to trigger a training run to produce a bespoke MLIP. This cycle transforms our system from a simple data generator into a true "apprentice" potential builder, setting the stage for the full automation of the active learning loop in the next cycle.

## 2. System Architecture

The architecture for CYCLE02 integrates the new Surrogate Explorer and Trainer modules into the existing pipeline. The data flow now includes a crucial filtering step, and a new branch is created for the training process. The file structure is expanded to include the new modules and their associated schemas and tests.

**File Structure for CYCLE02:**

The files and directories to be created or modified in this cycle are marked in bold. The structure builds directly upon the work from CYCLE01.

```
mlip_autopipec/
├── src/
│   ├── mlip_autopipec/
│   │   ├── __init__.py
│   │   ├── main.py             # CLI is updated with a 'train' command
│   │   ├── settings.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── user_config.py    # Updated to include training parameters
│   │   │   ├── system_config.py  # Updated with surrogate and trainer sections
│   │   │   ├── dft.py
│   │   │   └── **data.py**           # Schemas for database records (formalized)
│   │   ├── modules/
│   │   │   ├── __init__.py
│   │   │   ├── a_generator.py
│   │   │   ├── **b_explorer.py**     # Module B: Surrogate Explorer
│   │   │   ├── c_dft_factory.py
│   │   │   └── **d_trainer.py**      # Module D: Pacemaker Trainer
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── ase_utils.py
│   │       ├── qe_utils.py
│   │       └── **pacemaker_utils.py** # Utilities for Pacemaker
├── tests/
│   ├── __init__.py
│   ├── test_schemas.py
│   └── test_modules/
│       ├── __init__.py
│       ├── test_a_generator.py
│       ├── **test_b_explorer.py**
│       ├── test_c_dft_factory.py
│       └── **test_d_trainer.py**
├── pyproject.toml
└── README.md
```

**Architectural Blueprint:**

The workflow is now a sequential pipeline with an explicit training step at the end.

1.  The process begins as in CYCLE01, with the user providing a configuration that is expanded into a `SystemConfig` object.
2.  Module A (`a_generator.py`) generates a large number (`~10^3-10^4`) of candidate structures.
3.  **New Step**: This list of structures is now passed to **Module B (`b_explorer.py`)**.
4.  Inside Module B, each structure is first evaluated using a pre-trained MACE-MP model. The model calculates the energy and forces. Any structure that is physically nonsensical (e.g., atoms are too close, causing an extremely high energy) is discarded.
5.  The remaining structures are converted into a feature representation using a structural descriptor (e.g., SOAP or ACE).
6.  The Farthest Point Sampling (FPS) algorithm is then applied to this feature space. It greedily selects a small, user-defined number of structures (e.g., 200) that are maximally diverse.
7.  This highly-curated, smaller list of structures is then passed to Module C (`c_dft_factory.py`), which proceeds as before, running DFT calculations and storing the results in the ASE database.
8.  **New Step**: After the data generation is complete, the user can now invoke a new command, `mlip-auto train`.
9.  This command triggers **Module D (`d_trainer.py`)**.
10. The trainer reads all the data from the specified ASE database.
11. It uses `pacemaker_utils.py` to automatically generate a `pacemaker.in` configuration file. This utility will set parameters for the potential, such as the radial basis functions, the angular basis functions (body order), and the loss function weights for energy, forces, and stress. It will also specify the Delta Learning approach.
12. The trainer then invokes the `pacemaker` executable as a subprocess to perform the fitting.
13. Upon successful completion, the trained potential file (e.g., `FeNi_pot_v1.yace`) and the training logs are saved to a designated output directory.

This architecture inserts a powerful data-selection step that dramatically improves the efficiency of the pipeline and introduces the core ML functionality that the project is centered around.

## 3. Design Architecture

The design for CYCLE02 focuses on integrating the new modules through extensions to the existing Pydantic schema and creating well-defined interfaces for the new functionalities.

**Pydantic Schema Design:**

*   **`user_config.py` & `system_config.py`**: These schemas will be extended to control the new modules.
    *   A new `SurrogateConfig(BaseModel)` will be added. It will contain fields like `model_path: str` (for the MACE model), `num_to_select_fps: int`, and `descriptor_type: str` (e.g., 'SOAP').
    *   A new `TrainerConfig(BaseModel)` will be added. This will include parameters for the Pacemaker potential, such as `radial_basis: str`, `max_body_order: int`, and `loss_weights: Dict[str, float]`.
    *   These new models will be incorporated into the top-level `SystemConfig`, making them available to the entire pipeline.
    *   **Invariants**: `num_to_select_fps` must be less than the total number of generated structures. The keys in `loss_weights` must be one of `'energy'`, `'forces'`, `'stress'`.

*   **`data.py`**: The schema for database records will be formalized here.
    *   `StructureRecord(BaseModel)`: This will represent a single row in our database. It will include the `atoms: ase.Atoms` object itself, and metadata fields like `config_type: str`, `source: str` (e.g., `'initial_sqs'`, `'selected_by_fps'`), and potentially the `surrogate_energy: float` calculated by MACE.
    *   **Producers**: Modules A, B, and C all produce or modify these records.
    *   **Consumers**: Module D is the primary consumer of these records.

**Module and Utility Design:**

*   **Module B (`b_explorer.py`):**
    *   It will be instantiated with a `SurrogateConfig` object.
    *   It will need a `load_model()` method to initialize the MACE potential from its file.
    *   The core method, `select_structures(structures: List[ase.Atoms])`, will orchestrate the two-step process:
        1.  **Filtering**: Loop through structures, calculate MACE energy, and discard outliers.
        2.  **Sampling**: Convert the remaining structures to feature vectors. An external library like `dscribe` might be used for this. Then, implement the FPS algorithm. FPS is a simple greedy algorithm: start with a random structure, then iteratively add the structure that is farthest from the set of already-selected structures, until the desired number is reached.
    *   The method will return the final, smaller list of `ase.Atoms` objects.

*   **Module D (`d_trainer.py`):**
    *   It will be configured with a `TrainerConfig` object.
    *   The main method, `train_potential(database_path: str)`, will be the entry point.
    *   It will use ASE's `ase.db.connect()` to read all data from the database.
    *   It will call a utility function in `pacemaker_utils.py`, `generate_pacemaker_input(config: TrainerConfig, data: List[StructureRecord])`, which will be responsible for creating the `pacemaker.in` file string. This function will contain the logic for setting up the ACE basis, specifying the dataset, and defining the loss function. It will specifically configure the solver to use the Delta Learning approach with a ZBL baseline.
    *   The module will then use `subprocess.run` to execute the `pacemaker --train` command, pointing to the generated input file and the data. It will need to handle the conversion of the ASE database to the format Pacemaker expects.
    *   It will monitor the process and, upon completion, save the output potential and log files.

This design ensures that the new logic is encapsulated within its own modules and that the configuration is handled cleanly through the central schema, maintaining the modular and robust nature of the system.

## 4. Implementation Approach

The implementation will proceed by first updating the configuration, then developing the explorer and trainer modules, and finally updating the main CLI to incorporate the new functionality.

**Step 1: Update Schemas and Configuration**
*   Modify `schemas/user_config.py` and `schemas/system_config.py` to include the new `SurrogateConfig` and `TrainerConfig` Pydantic models.
*   Update the heuristic engine to populate these new configuration sections with sensible default values if they are not provided by the user.

**Step 2: Implement the Surrogate Explorer (Module B)**
*   Create `modules/b_explorer.py`.
*   Install necessary dependencies, including the MACE model (`mace-torch`) and a descriptor library like `dscribe`.
*   Implement the `select_structures` method.
*   For the MACE filtering part, load the pre-trained MACE-MP model. The ASE interface to MACE (`mace.calculators.mace_mp`) can be used here to easily calculate energies. A simple energy cutoff or a statistical outlier detection method can be used to filter structures.
*   For the FPS part, use `dscribe` to generate SOAP descriptors for each structure. Implement the greedy FPS algorithm logic.
*   Write unit tests in `tests/test_modules/test_b_explorer.py`. A key test will involve creating a simple, known set of 2D points, and asserting that the FPS algorithm correctly selects the points on the convex hull of the set. We will also mock the MACE model to test the filtering logic.

**Step 3: Implement the Pacemaker Trainer (Module D)**
*   Create `modules/d_trainer.py` and `utils/pacemaker_utils.py`.
*   In `pacemaker_utils.py`, write the `generate_pacemaker_input` function. This will be a templating function that takes the `TrainerConfig` and returns a formatted string for the `pacemaker.in` file. This is a critical piece of logic that translates our high-level configuration into the specific format required by the external tool.
*   In `d_trainer.py`, implement the `train_potential` method.
*   This method will first call the utility function to get the input file content.
*   It will then need to prepare the data. Pacemaker can read extended XYZ files. A utility function will be needed to convert the data from the ASE database into a single `.extxyz` file.
*   Finally, it will use `subprocess.run` to call the `pacemaker` executable. It will need to check the return code for success or failure and handle any errors.
*   Unit tests in `tests/test_modules/test_d_trainer.py` will test the input file generation logic. We will provide a sample `TrainerConfig` and assert that the generated string contains the correct keywords and values. We can mock `subprocess.run` to test the process execution logic.

**Step 4: Update the Main CLI**
*   Modify `main.py` to integrate the new modules into the data generation workflow. The main `run` command will now execute Module A, then Module B, then Module C in sequence.
*   Add a new command to the Typer CLI: `train`. This command will take a database path and a configuration file as input, instantiate the `Trainer`, and call its `train_potential` method.

This structured implementation approach allows us to build and test each new component before integrating it into the main workflow, simplifying development and debugging.

## 5. Test Strategy

The test strategy for CYCLE02 focuses on verifying the intelligence and efficiency introduced by the new modules. We need to ensure that the surrogate model is correctly filtering and selecting structures and that the trainer is capable of producing a valid potential.

**Unit Testing Approach (Min 300 words):**

Unit tests for CYCLE02 will be critical for validating the complex logic within the new modules.

*   **Surrogate Explorer (`test_b_explorer.py`):**
    *   **MACE Filtering:** We will test the filtering logic. We will create a list of `ase.Atoms` objects and mock the MACE calculator to return a specific list of energies, including some extreme outliers. We will then assert that the `select_structures` method correctly identifies and discards these outliers.
    *   **FPS Algorithm:** The correctness of the FPS algorithm is crucial. We will create a deterministic test case, for example, with a set of points forming a simple geometric shape (like an 'L' shape in 2D). We will provide this set to our FPS implementation and assert that it selects the most geometrically diverse points (the corners and endpoints of the 'L') in the correct order. This test will not involve any `ase.Atoms` objects but will test the raw algorithm's logic using NumPy arrays, ensuring its correctness before it's applied to real structural data.
    *   **Descriptor Generation:** We will test the integration with the `dscribe` library by creating a simple `ase.Atoms` object (e.g., a water molecule) and asserting that the generated SOAP descriptor has the correct shape and type.

*   **Trainer (`test_d_trainer.py`):**
    *   **Input File Generation:** This is the most important unit test for Module D. We will create a sample `TrainerConfig` Pydantic object with specific values for basis functions, body order, and loss weights. We will then call the `generate_pacemaker_input` utility function and assert that the returned string is a correctly formatted `pacemaker.in` file. We will use string matching and regular expressions to check that the correct keywords are present and are assigned the correct values from the config object. This ensures that our configuration is correctly translated for the external tool.
    *   **Data Preparation:** We will test the utility function that converts data from the ASE database to the `.extxyz` format that Pacemaker requires. We will create a temporary ASE database with a few known structures and then run the conversion. We will then parse the resulting `.extxyz` file to ensure it contains the correct atomic coordinates, energies, forces, and stresses in the correct format.

**Integration Testing Approach (Min 300 words):**

The integration test for CYCLE02 will verify the entire semi-automated workflow, from intelligent data selection to the final training of a potential. It will build on the Silicon test case from CYCLE01.

*   **Test Scenario: Building an MLIP for Silicon**
    *   **Objective**: To use the full pipeline to generate a dataset for Silicon, intelligently select a subset using the surrogate, and then train a basic MLIP that can accurately reproduce the Equation of State (EOS) curve.
    *   **Setup:** We will start with the same Silicon system from the CYCLE01 UAT.
        1.  First, we will run the `a_generator` to create a large number of strained and rattled silicon supercells (e.g., 1000 structures).
        2.  These 1000 structures will be fed into the `b_explorer`. We will configure it to use MACE to filter the structures and then use FPS to select the 50 most diverse structures.
        3.  These 50 structures will be passed to the `c_dft_factory` to get their ground-truth DFT energies and forces.
    *   **Execution:**
        1.  We will run the `mlip-auto run` command with a configuration that specifies the above generation and selection process. This tests the first half of the integration.
        2.  After this command completes, we will have a database containing only 50 DFT-calculated structures.
        3.  Next, we will run the `mlip-auto train` command, pointing it to the database we just created. This will trigger the Pacemaker training process and produce a `Si_pot_v1.yace` file.
    *   **Validation:**
        1.  First, we check that the number of structures in the final database is indeed 50, verifying that the surrogate selection worked.
        2.  The primary validation is to test the quality of the generated `Si_pot_v1.yace`. We will write a validation script that uses this new potential as an ASE calculator.
        3.  The script will calculate the energy of a range of strained Silicon supercells (the same range used to generate the EOS curve). This is a pure inference step; no DFT is involved.
        4.  We will plot the EOS curve predicted by our new MLIP and compare it to the original DFT data.
        5.  The test will pass if the MLIP's predicted equilibrium lattice constant is within 1% of the DFT value and if the Root Mean Squared Error (RMSE) of the energies across the entire curve is below a small threshold (e.g., < 5 meV/atom). This provides a quantitative measure of the potential's quality and successfully validates the entire CYCLE02 workflow.
