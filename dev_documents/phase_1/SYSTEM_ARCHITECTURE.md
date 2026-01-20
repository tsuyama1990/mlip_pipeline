# MLIP-AutoPipe System Architecture

## 1. Summary

The MLIP-AutoPipe project, titled the "Zero-Human" Protocol, is an ambitious initiative designed to completely automate the creation and refinement of Machine Learning Interatomic Potentials (MLIPs). In the field of materials science, a significant challenge exists, often termed the "chicken and egg" problem. To build a highly accurate MLIP, one requires a vast dataset of high-quality quantum mechanical calculations, typically from Density Functional Theory (DFT). However, generating this data efficiently requires effective sampling of the material's potential energy surface, a task best performed with a reasonably accurate potential. This paradox creates a bottleneck, demanding extensive human expertise and computational resources to bootstrap the process. Traditional methods, such as Ab Initio Molecular Dynamics (AIMD), are computationally expensive and often produce data with high time correlation, leading to inefficient learning.

MLIP-AutoPipe aims to solve this by creating a fully autonomous, closed-loop system that intelligently navigates the process from initial structure generation to active learning and production-scale simulation. The core philosophy is "Zero-Human" intervention. This means a user, even one without deep expertise in DFT or machine learning, can initiate a workflow by providing only a minimal configuration: the chemical elements involved and the desired physical properties. The system then takes over, managing the complex, multi-stage process of data generation, model training, and simulation without requiring further input.

To achieve this ambitious goal, the system is built upon three core technological strategies. First is the **Surrogate-First Strategy**, which employs a pre-trained, general-purpose foundation model (like MACE-MP) as an inexpensive "scout." Instead of immediately running costly DFT calculations on initial candidate structures, the system uses the surrogate model to perform a rapid pre-screening, eliminating physically implausible structures and identifying promising candidates. This allows the system to explore a vast configuration space at a fraction of the cost. Second is the principle of **Decoupled Inference and Training**. The production simulation (e.g., Molecular Dynamics) does not halt while waiting for new training data to be processed. When the simulation engine encounters a structure with high uncertainty, it is added to a queue. A separate, asynchronous training engine processes this queue in the background, updating the MLIP, which is then seamlessly swapped into the simulation engine. This ensures that the primary simulation progresses without interruption. The third and most innovative strategy is **Periodic Embedding with Force Masking**. When extracting a local atomic environment for DFT calculation, the system avoids creating artificial surfaces by cutting a small, periodic box from the larger simulation cell. Crucially, it then "masks" the forces on the atoms near the boundary of this box, preventing the learning algorithm from being corrupted by artificial boundary effects. This ensures the model learns the true bulk properties of the material, a critical factor for achieving high fidelity.

## 2. System Design Objectives

The primary objective of the MLIP-AutoPipe system is to establish a fully autonomous, reliable, and efficient workflow for generating bespoke MLIPs. The success of this project is predicated on achieving a set of clearly defined goals, while operating within specific constraints.

**Goals:**
1.  **Full Autonomy ("Zero-Human"):** The system must be capable of managing the entire MLIP generation lifecycle without human intervention. This includes initial structure generation, DFT parameter selection, calculation execution, error recovery, model training, and active learning. The only required user input should be a high-level definition of the material and the simulation goal.
2.  **High-Fidelity Potentials:** The ultimate output must be a machine learning potential that accurately reproduces the forces, energies, and stresses predicted by DFT. The potential should be robust enough for use in large-scale, long-duration simulations to predict meaningful physical properties, such as phase diagrams or diffusion coefficients.
3.  **Computational Efficiency:** The system must be designed to minimise the use of expensive DFT calculations. This will be achieved through intelligent sampling strategies, such as the surrogate-first approach and active learning, ensuring that computational effort is focused only on the most informative atomic configurations.
4.  **Robustness and Fault Tolerance:** The workflow involves running thousands of external DFT calculations, which can fail for numerous reasons. The system must be highly resilient, with built-in automated error detection and recovery mechanisms. It must also support checkpointing, allowing a long-running workflow to be resumed seamlessly after an interruption.
5.  **Modularity and Extensibility:** The architecture must be modular, allowing individual components (like the DFT engine or MLIP framework) to be updated or replaced without redesigning the entire system. For instance, it should be straightforward to add support for a different DFT code or a new machine learning model.

**Constraints:**
1.  **Open-Source Stack:** The entire system will be built using freely available, open-source software to ensure accessibility and encourage community adoption. This includes Python as the core language, and established scientific libraries like ASE, Pymatgen, Quantum Espresso, LAMMPS, and Pacemaker.
2.  **HPC Environment:** The system is designed to operate on High-Performance Computing (HPC) clusters, leveraging schedulers like Slurm or PBS to manage parallel execution of numerous DFT and MD jobs.
3.  **Minimal User Configuration:** The system design must adhere to the principle of minimal user input. It must derive all necessary low-level parameters (e.g., DFT convergence settings, k-point meshes) from a small set of high-level user directives.

**Success Criteria:**
The success of the MLIP-AutoPipe system will be measured by its ability to:
-   Successfully generate a converged MLIP from only a `input.yaml` file, without any code modifications or manual interventions.
-   Demonstrate a significant reduction (e.g., >50%) in the number of DFT calculations required to reach a target accuracy compared to a traditional AIMD-based workflow.
-   Produce a potential that can predict a target physical property (e.g., the melting point of a material or the diffusion coefficient of an alloy) within a specified tolerance (e.g., 10%) of a reference value from literature or direct DFT simulation.
-   Successfully recover from at least three different common DFT convergence failure scenarios without crashing the entire workflow.

## 3. System Architecture

The MLIP-AutoPipe system is designed as a multi-phase, asynchronous pipeline that coordinates a series of specialised modules. Each module has a single responsibility, and they communicate through a central database and a task queue. This decoupled design ensures robustness and scalability. The overall data flow can be conceptually divided into three main phases: Cold Start, Training Loop, and Production.

```mermaid
graph TD
    subgraph Legend
        direction LR
        User_Input((User Input))
        System_Component[System Component]
        Data_Store[(Data Store)]
        External_Process[External Process]
    end

    subgraph Phase 1: Cold Start - Seeding the Workflow
        User_Input[User: Minimal YAML Config] --> FullConfig[System: Heuristic Config Engine]
        FullConfig --> A[Module A: Physics-Informed Generator]
        A -- Diverse Structures --> B[Module B: Surrogate Explorer]
        B -- Pre-filtered Structures --> C[Selector: Farthest Point Sampling]
    end

    subgraph Phase 2: Training Loop - The DFT Factory
        C -- Info-rich Structures --> DFT_Queue{Task Queue}
        F -- High-Uncertainty Structures --> DFT_Queue
        DFT_Queue --> D[Module C: Automated DFT Factory]
        D -- Calculation --> QE[External: Quantum Espresso]
        QE -- Error --> D
        D -- Success --> DB[(ASE Database)]
        DB --> E[Module D: Pacemaker Trainer]
        E -- Train --> Pacemaker[External: Pacemaker]
        Pacemaker -- Updated Potential --> Active_Potential[Active Potential File (.yace)]
    end

    subgraph Phase 3: Production - The Explorer
        Active_Potential --> F[Module E: MD/kMC Inference Engine]
        F -- Simulation --> LAMMPS[External: LAMMPS]
        F -- Uncertainty OK --> Analysis[Physical Properties Analysis]
    end

    style User_Input fill:#f9f,stroke:#333,stroke-width:2px
    style Data_Store fill:#ccf,stroke:#333,stroke-width:2px
```

**Phase 1: Cold Start (The Seed)**
The workflow begins with a minimal user configuration file. The Heuristic Engine expands this into a detailed execution plan, determining all necessary DFT parameters. **Module A (Physics-Informed Generator)** then creates an initial, diverse set of atomic structures without any DFT calculations. It uses established techniques like Special Quasirandom Structures (SQS) for alloys or Normal Mode Sampling for molecules. This initial batch is passed to **Module B (Surrogate Explorer)**, which uses a pre-trained universal potential (MACE) to quickly evaluate forces and energies, discarding any unstable configurations. Finally, the **Farthest Point Sampling (FPS) Selector** computes structural fingerprints for the remaining candidates and selects a small, maximally diverse subset to send for actual DFT calculation. This ensures the initial training set provides the broadest possible coverage of the configuration space.

**Phase 2: Training Loop (The Factory)**
The selected structures are placed into a **Task Queue**. This queue feeds **Module C (The Automated DFT Factory)**, which is the robust heart of the system. This module is a wrapper around a DFT engine (Quantum Espresso) that manages job submission, parameter validation, and, crucially, automated error recovery. If a calculation fails to converge, the module intelligently adjusts parameters (e.g., electronic mixing settings) and retries. Successful calculations (energy, forces, stress) are stored in a central **ASE Database**. **Module D (Pacemaker Trainer)** continuously monitors the database. When enough new data is available, it automatically triggers a training job using the Pacemaker framework. The output is an updated MLIP file, which becomes the new "active" potential for the production phase.

**Phase 3: Production (The Explorer)**
**Module E (MD/kMC Inference Engine)** uses the latest active potential to run large-scale simulations with LAMMPS. Its primary goal is to explore the potential energy surface to find regions where the model is uncertain. It uses the potential's built-in `extrapolation_grade` as a measure of uncertainty. If the uncertainty remains below a threshold, the simulation continues, generating data for calculating physical properties. If the uncertainty exceeds the threshold, the simulation triggers the **Periodic Embedding** logic. It extracts a small, periodic sub-system around the uncertain atom(s) and sends this new structure back to the DFT Task Queue. This closes the active learning loop, constantly feeding the model new, informative data from regions it has not yet learned well. This decoupled, asynchronous loop allows the production simulation to run continuously while the potential is refined in the background.

## 4. Design Architecture

The software architecture is designed to be modular, testable, and maintainable, with a strict separation of concerns. The entire system is built around a schema-first philosophy, using Pydantic for all data models to ensure data integrity at the boundaries of each component.

**File Structure:**
The proposed file structure clearly separates configuration, source code, tests, and documentation.

```
mlip-autopipe/
├── .github/              # CI/CD workflows
├── dev_documents/
│   ├── ALL_SPEC.md
│   └── system_prompts/   # All generated documentation
│       ├── SYSTEM_ARCHITECTURE.md
│       └── CYCLE01/ ... CYCLE08/
├── mlip_autopipec/          # Main source code package
│   ├── __init__.py
│   ├── app.py              # CLI entry point (Typer)
│   ├── workflow_manager.py # Main orchestrator class
│   ├── config/
│   │   ├── __init__.py
│   │   └── models.py       # Pydantic models for all configuration
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── generation.py   # Module A: Physics-Informed Generator
│   │   ├── exploration.py  # Module B: Surrogate Explorer
│   │   ├── dft.py          # Module C: Automated DFT Factory
│   │   ├── training.py     # Module D: Pacemaker Trainer
│   │   └── inference.py    # Module E: MD/kMC Inference Engine
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── ase_utils.py    # Helpers for ASE database interactions
│   │   └── resilience.py   # Decorators for retry logic, error handling
│   └── data/
│       └── sss_p_data.json # Data for SSSP pseudopotentials
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   └── modules/
│       ├── test_generation.py
│       └── test_dft.py
├── pyproject.toml        # Project metadata and dependencies (uv)
└── README.md
```

**Class/Function Overview:**
-   **`app.py`**: A thin wrapper using Typer to provide a user-friendly Command Line Interface. It will parse the user's `input.yaml` and delegate control to the `WorkflowManager`.
-   **`WorkflowManager`**: The central orchestrator. It holds the state of the workflow, initialises all the modules, and manages the main active learning loop.
-   **`config/models.py`**: Contains all Pydantic models. This is the single source of truth for the project's data structures, including `UserInputConfig`, `SystemConfig`, `DFTCalculation`, and `TrainingJob`. Strict validation rules will be enforced here.
-   **`modules/generation.py`**: `StructureGenerator` class. Takes system configuration and produces a list of initial `ase.Atoms` objects based on the chosen strategy (SQS, NMS, etc.).
-   **`modules/dft.py`**: `DFTFactory` class. Manages all interactions with the external DFT code. It contains methods like `run_calculation()` and the `auto_recovery()` logic for handling convergence failures. It takes an `ase.Atoms` object and returns DFT results.
-   **`modules/training.py`**: `PacemakerTrainer` class. This class is responsible for monitoring the database, generating the necessary input files for the Pacemaker code, launching the training process, and handling the resulting potential file.
-   **`modules/inference.py`**: `LammpsRunner` class. It configures and runs LAMMPS simulations using the current best potential. It includes the logic to monitor uncertainty on-the-fly.
-   **`utils/resilience.py`**: Contains generic decorators, such as `@retry_on_failure`, which can be applied to methods in the `DFTFactory` to implement the auto-recovery logic cleanly, without cluttering the main code.

**Data Models:**
The use of Pydantic is central to the design. For example, the `DFTCalculation` model will not only define the fields (energy, forces, stress) but also include validators to ensure that, for instance, the `forces` array has the correct shape `(N, 3)` where `N` is the number of atoms. The `UserInputConfig` will have validators to ensure the composition percentages sum to 1.0. This schema-first approach prevents a large class of bugs by catching invalid data at the earliest possible stage, making the whole system more robust and easier to debug.

## 5. Implementation Plan

The project will be developed over eight sequential cycles, each building upon the last. This iterative approach allows for testing and validation at each stage before adding more complexity.

**Cycle 1: The Foundation - Automated DFT Factory**
This cycle focuses on creating the most critical component: a robust and automated DFT calculation engine. The goal is to create a Python class that can take an `ase.Atoms` object and reliably return its DFT energy, forces, and stress, including handling common errors.
-   **Features:**
    -   Develop a `DFTFactory` class that wraps Quantum Espresso.
    -   Implement the heuristic logic for automatically determining key DFT parameters (cutoffs from SSSP, k-points based on cell size, smearing for metals).
    -   Implement the auto-recovery logic for at least two common convergence errors (e.g., by reducing `mixing_beta`).
    -   Create a simple database utility to store the results of successful calculations.
-   **Outcome:** A command-line tool that can reliably run a DFT calculation on a single atomic structure file and store the result.

**Cycle 2: Physics-Informed Initial Structure Generation**
With the DFT engine in place, the next step is to generate the initial data. This cycle implements Module A.
-   **Features:**
    -   Implement the SQS generator for alloys, including applying strain and rattling.
    -   Implement the NMS generator for molecules.
    -   Implement the Melt-Quench and Defect Engineering strategies for crystals.
    -   Integrate this module with the DFT Factory from Cycle 1 to create a pipeline that can generate and calculate an initial training set.
-   **Outcome:** A script that can generate a diverse set of hundreds of structures and populate the database with their DFT results.

**Cycle 3: Surrogate-First Exploration and Smart Selection**
This cycle aims to make the initial data generation more efficient by integrating the surrogate model and FPS selector (Module B).
-   **Features:**
    -   Integrate the MACE-MP model to perform fast pre-calculation screening on generated structures.
    -   Implement the Farthest Point Sampling (FPS) algorithm using a suitable structural fingerprint (e.g., SOAP).
    -   Create a workflow where thousands of structures are generated, filtered by MACE, and then down-selected to a few hundred by FPS before being sent to the DFT Factory.
-   **Outcome:** A significantly more efficient initial data generation pipeline that selects more informative structures for the same computational cost.

**Cycle 4: The Training Engine - Closing the First Loop**
This cycle introduces the MLIP training component (Module D), creating the first, non-active learning loop.
-   **Features:**
    -   Develop a `PacemakerTrainer` class that can read data from the database.
    -   Automate the generation of Pacemaker input files.
    -   Implement the logic to run the Pacemaker training process and store the output potential file.
    -   Implement Delta Learning and set default loss weights.
-   **Outcome:** A workflow that can go from initial structure generation to a fully trained MLIP.

**Cycle 5: The OTF Inference Engine - Live Simulation**
This cycle implements the production simulation component (Module E), enabling On-The-Fly (OTF) dynamics.
-   **Features:**
    -   Develop a `LammpsRunner` class to run MD simulations using a trained potential from Cycle 4.
    -   Implement the logic to monitor the `extrapolation_grade` during the simulation.
    -   Define the threshold logic for identifying when a structure is "uncertain."
-   **Outcome:** A system that can run a stable MD simulation and detect when it encounters an atomic configuration that is outside its training data.

**Cycle 6: Intelligent Data Extraction - The Active Learning Loop**
This is a critical cycle that fully closes the active learning loop by implementing the periodic embedding strategy.
-   **Features:**
    -   Implement the Periodic Embedding algorithm to extract small, periodic sub-systems from the larger MD simulation around uncertain atoms.
    -   Implement the Force Masking logic to set the weights of buffer atoms to zero.
    -   Integrate this output with the DFT Factory's task queue.
-   **Outcome:** A complete, closed-loop active learning system that can autonomously improve its own potential by exploring new configurations during a live simulation.

**Cycle 7: Configuration, Data Management, and CLI**
This cycle focuses on usability and robustness by finalising the data models and user interface.
-   **Features:**
    -   Develop strict Pydantic models for all user and system configurations.
    -   Finalise the extended ASE database schema with all required metadata.
    -   Build a polished Command Line Interface (CLI) using Typer (e.g., `mlip-auto run input.yaml`).
-   **Outcome:** A user-friendly, robust, and fully configurable application.

**Cycle 8: Monitoring, Usability, and Release**
The final cycle adds monitoring capabilities and prepares the project for release.
-   **Features:**
    -   Develop a simple web-based dashboard (e.g., using Streamlit or Dash) to visualise the workflow's progress, including the training set size, model RMSE over time, and the uncertainty histogram.
    -   Write comprehensive user documentation and tutorials.
    -   Perform final stress testing on the entire workflow.
-   **Outcome:** A polished, well-documented, and user-friendly system ready for its initial release.

## 6. Test Strategy

Each implementation cycle will be accompanied by a rigorous, multi-layered testing strategy to ensure correctness, robustness, and maintainability.

**Cycle 1: DFT Factory**
-   **Unit Tests:** Verify the heuristic logic in isolation. For example, provide a structure with known lattice vectors and assert that the generated k-point mesh is correct. Test the auto-recovery mechanism by mocking a DFT failure and asserting that the `DFTFactory` retries with the correct modified parameters.
-   **Integration Tests:** Run tests against a real Quantum Espresso installation using a small, simple system (e.g., a 2-atom silicon cell). Verify that the end-to-end process—from `ase.Atoms` object to a result stored in the database—completes successfully and the calculated energy is close to a known reference value.

**Cycle 2: Structure Generation**
-   **Unit Tests:** For the SQS generator, assert that the output structure has the correct composition and number of atoms. For the defect generator, assert that a vacancy structure truly has one fewer atom than the original. Test edge cases, like a request for 100% composition of a single element.
-   **Integration Tests:** Test the full pipeline: generate a structure, then immediately feed it to the DFT Factory from Cycle 1, and verify the process completes without errors.

**Cycle 3: Surrogate & FPS**
-   **Unit Tests:** Test the FPS algorithm with a known set of vectors to ensure it selects the correct "farthest" points. Test the MACE filter to ensure it correctly identifies and removes structures with, for example, unphysically close atoms.
-   **Integration Tests:** Create a large set of mock structures, run the entire selection pipeline (generation -> MACE filter -> FPS), and assert that the final number of selected structures is within an expected range and significantly smaller than the initial set.

**Cycle 4: Training Engine**
-   **Unit Tests:** Test the Pacemaker input file generation logic. Create a mock dataset in the database and assert that the generated input file has the correct format and contains the right number of structures.
-   **Integration Tests:** Run a mini-training loop with a small, pre-calculated dataset. Assert that the training process completes successfully and generates a valid `.yace` potential file. This test will be computationally expensive and may be run less frequently.

**Cycle 5: Inference Engine**
-   **Unit Tests:** Test the uncertainty monitoring logic. Create a mock potential that returns a known uncertainty for a given structure and assert that the `LammpsRunner` correctly flags it as uncertain when the threshold is crossed.
-   **Integration Tests:** Run a short LAMMPS simulation with a real potential file from Cycle 4. Verify that the simulation starts, runs for a few steps, and terminates cleanly.

**Cycle 6: Active Learning Loop**
-   **Unit Tests:** Test the Periodic Embedding logic with a large `ase.Atoms` object. Assert that the extracted sub-system has the correct dimensions, is periodic, and contains the correct atoms. Test the Force Masking logic to ensure the mask array correctly identifies core and buffer atoms.
-   **Integration Tests:** This is the most complex integration test. It will involve a mock MD simulation that deliberately produces a high-uncertainty structure, triggering the embedding, and asserting that a new calculation for the small, embedded cell appears in the DFT queue.

**Cycle 7: Configuration and CLI**
-   **Unit Tests:** Test the Pydantic models extensively. Provide invalid configurations (e.g., compositions not summing to 1.0, invalid element symbols) and assert that `ValidationError` is raised.
-   **Integration Tests:** Use the Python `subprocess` module to test the CLI entry point. Run the command with various valid and invalid `input.yaml` files and check for correct exit codes and error messages.

**Cycle 8: Monitoring and Release**
-   **Unit Tests:** Test the data-parsing functions for the dashboard to ensure they correctly calculate metrics like RMSE from the database.
-   **End-to-End Tests:** The final test will be a full "bake-off." Run the entire workflow for a known, simple material (e.g., bulk silicon) and assert that it produces a potential that can predict the lattice constant and bulk modulus to within 5% of the accepted values. This will serve as the final validation of the entire system.
