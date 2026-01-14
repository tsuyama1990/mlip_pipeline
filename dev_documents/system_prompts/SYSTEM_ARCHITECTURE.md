# MLIP-AutoPipe System Architecture

- **Project Name**: MLIP-AutoPipe (The "Zero-Human" Protocol)
- **Version**: 1.0.0
- **Author**: System Architect (AI Assistant)
- **Target Audience**: Backend Engineers, Materials Scientists, HPC Administrators

---

## 1. Summary

The field of materials science is constrained by a fundamental conflict: the deep, predictive accuracy of first-principles quantum mechanical calculations (like Density Functional Theory, DFT) is computationally expensive, limiting its application to small systems and short timescales. Conversely, classical methods like Molecular Dynamics (MD) can simulate millions of atoms over long durations but rely on empirical potentials that often lack the accuracy to model complex chemical processes. Machine Learning Interatomic Potentials (MLIPs) have emerged as a revolutionary solution, promising the accuracy of DFT at a fraction of the computational cost. However, the development of a high-quality MLIP is a significant challenge in itself. It is a classic "chicken and egg" problem: creating a robust potential requires a vast and diverse dataset of atomic structures and their corresponding energies, forces, and stresses, but generating such a dataset efficiently requires a reliable potential to explore the configuration space in the first place.

This reliance on expert-driven, manually intensive data generation campaigns, often involving computationally expensive Ab Initio Molecular Dynamics (AIMD), creates a major bottleneck. AIMD explores the potential energy surface inefficiently, often generating highly correlated and redundant data, which provides diminishing returns for the immense computational investment. The MLIP-AutoPipe system is designed to shatter this paradigm. Its core mission is to establish a **"Zero-Human" protocol**, a fully autonomous workflow that orchestrates the entire lifecycle of an MLIP, from initial data generation to active learning and large-scale production simulations. The system is engineered to allow a user, even one without deep expertise in DFT or machine learning, to initiate a sophisticated materials science campaign by specifying only the constituent elements and the scientific goal.

To achieve this, MLIP-AutoPipe is built upon three core technological strategies. First is the **Surrogate-First Strategy**, which leverages large, pre-trained foundation models (like MACE-MP) as an initial, inexpensive "scout." Instead of starting with costly DFT calculations, the system generates tens of thousands of physically plausible candidate structures and uses the surrogate model to perform a rapid, low-cost assessment, filtering out unphysical configurations and identifying information-rich candidates. Second is the principle of **Decoupled Inference and Training**. The system separates the production simulation (e.g., an MD run) from the MLIP training process. When the simulation engine encounters a configuration where the potential's prediction is uncertain, it does not halt. Instead, it flags the structure, extracts the relevant local environment, and places it into a queue. A separate, asynchronous training engine processes this queue, performs the necessary DFT calculations, and retrains the potential. The updated potential is then seamlessly integrated back into the production simulation. This asynchronous, on-the-fly (OTF) approach maximizes simulation throughput. The third and most advanced strategy is **Periodic Embedding with Force Masking**. To avoid surface artefacts inherent in traditional cluster-based data generation, the system extracts local atomic environments as small, fully periodic cells. Crucially, it then "masks" the forces on the atoms in the buffer region of these cells, ensuring that the learning process only uses data from the core region, which accurately represents the bulk material. This novel technique significantly improves the quality and transferability of the learned potential. By integrating these strategies, MLIP-AutoPipe transforms MLIP generation from a bespoke, artisanal process into a systematic, automated, and highly efficient manufacturing pipeline for discovering new materials and their properties.

---

## 2. System Design Objectives

The primary objective of the MLIP-AutoPipe project is to engineer a fully autonomous, robust, and efficient system for the creation and deployment of Machine Learning Interatomic Potentials. The design is guided by a set of clear goals, constraints, and success criteria to ensure it meets the needs of both novice users and expert researchers.

**Goals:**

1.  **Full Automation ("Zero-Human" Protocol):** The system's foremost goal is to eliminate the need for manual intervention at every stage of the MLIP lifecycle. This includes automated initial structure generation, intelligent data selection for DFT calculations, heuristic-driven DFT parameterisation, error recovery, model retraining, and the initiation of production simulations. The user's role should be limited to providing a high-level definition of the material system and the desired scientific outcome.
2.  **Computational Efficiency:** To overcome the limitations of traditional AIMD-based data generation, the system must be exceptionally efficient. This will be achieved by minimizing the number of expensive DFT calculations performed. The Surrogate-First strategy is central to this goal, using fast, pre-trained models to vet millions of potential structures before committing DFT resources.
3.  **DFT-Level Accuracy:** The ultimate purpose of the MLIP is to serve as a surrogate for DFT. Therefore, the system must produce potentials that can predict energies, forces, and stresses with an accuracy that is statistically indistinguishable from the underlying DFT engine for the target application. The active learning loop is the key mechanism to achieve this, iteratively refining the model by adding data from configurations where the model shows the highest uncertainty.
4.  **Scalability and Robustness:** The system is intended for use in high-performance computing (HPC) environments. It must be designed to manage thousands of concurrent DFT calculations and large-scale MD simulations. This necessitates a scalable architecture, likely based on a task queue (like Dask or Celery), and robust error handling. The automated DFT factory must be capable of recovering from common failures (e.g., SCF convergence issues) by intelligently adjusting calculation parameters and retrying.
5.  **Accessibility and Usability:** The system should democratise access to high-fidelity materials simulation. A user with a materials science background but no specific expertise in DFT or ML should be able to use the system effectively. This is achieved through a minimal, declarative user input format (a simple YAML file) that abstracts away the complexities of the underlying simulation engines.

**Constraints:**

1.  **Technology Stack:** The project will adhere to a pre-defined technology stack: Python 3.11+, `uv` for environment management, `Pydantic` for data validation, `Pacemaker` for the core MLIP framework, `Quantum Espresso` as the DFT engine, `LAMMPS` as the MD engine, and `MACE` as the surrogate model. All development must be compatible with these choices.
2.  **Cold-Start Capability:** The system must be able to function without any pre-existing DFT data for the target material system. The Physics-Informed Generator module is explicitly designed to address this "cold-start" problem.
3.  **Static DFT Calculations:** The active learning database will be built exclusively from single-point "static" DFT calculations (`calculation = 'scf'`). The system will not perform structural relaxations within the DFT engine, as the entire point is to learn the forces on specific, potentially high-energy, configurations.

**Success Criteria:**

1.  **End-to-End Autonomous Operation:** The primary success criterion is the successful completion of an entire workflow for a non-trivial test case (e.g., a binary alloy like FeNi). The system, given only a minimal `input.yaml`, must autonomously generate an initial dataset, train a preliminary potential, run an MD simulation, identify uncertain structures, augment the dataset via the active learning loop, and produce a refined potential.
2.  **Verifiable Accuracy Improvement:** It must be demonstrably clear that the active learning loop improves the quality of the MLIP. This will be measured by tracking the Root Mean Square Error (RMSE) of energy, force, and stress predictions on a hold-out test set as the training dataset grows with each cycle.
3.  **Robustness in Practice:** The system must successfully handle at least three common, injected DFT failure modes (e.g., a convergence failure, a timeout) and recover automatically without crashing the entire workflow.
4.  **Comparable Physics Output:** The final, trained MLIP should be capable of calculating a physical property (e.g., the vacancy diffusion barrier in Ni) that is in reasonable agreement (e.g., within 15-20%) with established literature values or direct DFT calculations. This validates that the potential is not just accurate but also physically meaningful.

---

## 3. System Architecture

The MLIP-AutoPipe architecture is designed as a modular, multi-phase pipeline that systematically transforms a high-level user request into a high-fidelity, production-ready Machine Learning Interatomic Potential. The system is logically divided into three primary phases: **Phase 1: Cold Start (The Seed)**, **Phase 2: Training Loop (The Factory)**, and **Phase 3: Production (The Explorer)**. Each phase consists of several interconnected modules that perform specific tasks, communicating through a central database and a system of job queues.

```mermaid
graph TD
    User[User: Minimal Config] --> Heuristic[Heuristic Engine]
    Heuristic --> FullConfig[System: Full Execution Config]

    subgraph "Phase 1: Cold Start (The Seed)"
        FullConfig --> Generator[Module A: Physics-Informed Generator]
        Generator -- SQS/NMS/Melt --> Surrogate[Module B: MACE Surrogate]
        Surrogate --> FPS[Selector: Farthest Point Sampling]
    end

    subgraph "Phase 2: The Training Loop (The Factory)"
        FPS --> DFT_Queue
        Embed_Queue --> DFT_Queue
        DFT_Queue --> QE[Module C: Automated DFT Factory]
        QE -- Error --> Recovery[Auto-Recovery Logic]
        Recovery --> QE
        QE --> DB[(ASE Database: .extxyz)]
        DB --> Pacemaker[Module D: Pacemaker Trainer]
        Pacemaker --> Potential[Active Potential (.yace)]
    end

    subgraph "Phase 3: Production (The Explorer)"
        Potential --> Inference[Module E: MD/kMC Engine]
        Inference -- Uncertainty > Threshold --> Extractor[Periodic Embedding]
        Extractor --> Embed_Queue[Priority Queue]
        Inference -- Uncertainty OK --> Analysis[Physical Properties]
    end
```

**Module Descriptions:**

*   **Module A: Physics-Informed Generator:** This is the starting point of the pipeline, addressing the "cold-start" problem. It operates without any DFT data. Based on the material type specified by the user (alloy, molecule, crystal), it employs a suite of physics-based techniques to generate an initial, diverse set of atomic structures. For alloys, it uses Special Quasirandom Structures (SQS) combined with lattice strain and atomic rattling to sample different chemical orderings and elastic conditions. For molecules, it uses Normal Mode Sampling to explore the vibrational space efficiently. For crystals, it uses a surrogate-driven Melt-Quench procedure to create amorphous and liquid-phase structures, alongside systematic defect generation (vacancies, interstitials). The output of this module is a set of thousands of candidate structures designed to cover a wide range of physical configurations.

*   **Module B: Surrogate Explorer:** This module acts as an intelligent, cost-saving filter. It takes the large set of candidate structures from Module A and evaluates them using a fast, pre-trained universal potential (e.g., MACE-MP). This rapid pre-screening eliminates structures that are physically nonsensical (e.g., with overlapping atoms) without wasting expensive DFT cycles. Following this initial filtering, the module calculates a structural fingerprint (e.g., SOAP descriptor) for the remaining candidates. It then employs the Farthest Point Sampling (FPS) algorithm to select a subset of structures that are maximally diverse in descriptor space. This ensures the initial DFT calculations are as information-rich as possible.

*   **Module C: Automated DFT Factory:** This is the computational core of the system. It is a highly robust and automated wrapper around the Quantum Espresso DFT engine. Its sole purpose is to take a structure from the DFT queue and reliably calculate its energy, forces, and stress tensor. It operates purely in a static (`scf`) mode. Key features include a heuristic engine for automatically determining critical DFT parameters like pseudopotentials (via the SSSP library), k-point density, smearing for metals, and spin polarization for magnetic systems. Crucially, it contains an auto-recovery logic that can detect common QE failures (e.g., lack of SCF convergence) and automatically retry the calculation with adjusted, more lenient parameters. This resilience is vital for unattended, long-running workflows.

*   **Module D: Pacemaker Trainer:** This module is responsible for the "learning" part of the pipeline. It continuously monitors the central ASE database for new DFT results. When a sufficient number of new data points have been added, it automatically initiates a training job using the Pacemaker framework. It handles the generation of the necessary input files, specifies the training parameters (like loss weights for energy, forces, and stress), and executes the training process. The output is an updated and improved MLIP file (e.g., a `.yace` file for ACE potentials), which becomes the new "active potential" for the production phase.

*   **Module E: Scalable Inference & OTF (On-the-Fly) Engine:** This module uses the latest trained potential to run large-scale production simulations, typically Molecular Dynamics (MD) or kinetic Monte Carlo (kMC), using an engine like LAMMPS. While simulating, it constantly evaluates the uncertainty of the potential for the current atomic configuration using Pacemaker's built-in extrapolation grade metric. If the uncertainty exceeds a defined threshold, it signals that the simulation has entered an unknown region of the configuration space. At this point, the **Periodic Embedding** logic is triggered. It extracts the local atomic environment around the uncertain atom as a small, self-contained periodic cell and submits it to the `Embed_Queue` for DFT calculation. This closes the active learning loop. The simulation itself continues, perhaps with a smaller timestep, ensuring maximum throughput. This module is also responsible for the advanced **Force Masking** technique, which is critical for learning from these embedded periodic cells.

---

## 4. Design Architecture

The design of MLIP-AutoPipe is founded on the principles of modularity, testability, and configuration-driven execution. A strict schema-first development process, centered on Pydantic models, ensures that the complex interplay between modules is robust, validated, and easy to reason about. The architecture enforces a clear separation of concerns, from data modelling to process execution.

**File and Directory Structure:**

A logical file structure is adopted to separate concerns and facilitate maintainability. All core source code resides within the `src/` directory.

```
.
├── dev_documents/
├── pyproject.toml
├── src/
│   └── mlip_autopipec/
│       ├── __init__.py
│       ├── app.py              # Main CLI entry point using Typer
│       ├── config/
│       │   ├── __init__.py
│       │   ├── user.py         # Schema for the minimal user input (input.yaml)
│       │   └── system.py       # Schema for the fully-expanded system configuration
│       ├── modules/
│       │   ├── __init__.py
│       │   ├── generator.py    # Module A: PhysicsInformedGenerator class
│       │   ├── explorer.py     # Module B: SurrogateExplorer class
│       │   ├── dft_factory.py  # Module C: QEProcessRunner class
│       │   ├── trainer.py      # Module D: PacemakerTrainer class
│       │   └── inference.py    # Module E: LammpsRunner and Embedding logic
│       ├── data/
│       │   ├── __init__.py
│       │   └── database.py     # Wrapper for ASE DB with custom metadata handling
│       └── utils/
│           ├── __init__.py
│           └── logging.py      # Centralised logging configuration
└── tests/
    ├── conftest.py
    ├── config/
    │   └── test_schemas.py
    └── modules/
        └── test_dft_factory.py
```

**Data Models (Pydantic-based Schema-First Development):**

The entire system is orchestrated by a central configuration object, which is a validated Pydantic model. This approach provides several key advantages:
1.  **Validation:** Input data, both from the user and between internal modules, is strictly validated at runtime, preventing a wide class of errors.
2.  **Clarity:** The schemas serve as the "single source of truth" for the system's data structures, making the code self-documenting.
3.  **Decoupling:** Modules are initialized with the same global configuration object, ensuring consistent behaviour without hard-coding parameters.

*   `UserConfig (` `user.py` `)`: This model defines the minimal set of inputs a user must provide. It focuses on *what* the user wants (elements, composition, goal) rather than *how* to achieve it. It includes fields like `target_system`, `simulation_goal`, and `resources`.

*   `SystemConfig (` `system.py` `)`: This is the comprehensive, internal configuration model. A dedicated "Heuristic Engine" will take the `UserConfig` as input and expand it into a `SystemConfig` object. This process fills in hundreds of default parameters, from DFT convergence thresholds to MD simulation timesteps. For example, based on the user's `elements: ["Fe", "Ni"]`, the heuristic engine will automatically populate the `SystemConfig` with the correct SSSP pseudopotentials, recommended wavefunction cutoffs, and instructions to enable spin polarization. This powerful model is then passed to all modules, providing them with all the necessary parameters to perform their tasks.

**Core Classes and Components:**

The logic is encapsulated within classes inside their respective modules.

*   `QEProcessRunner` (`dft_factory.py`): This class is not just a simple command-line wrapper. It is responsible for taking an ASE `Atoms` object and the `SystemConfig`, generating a complete Quantum Espresso input file, executing the `pw.x` binary, parsing the output to extract energy/forces/stress, and handling errors. Its auto-recovery logic will be implemented as a state machine that modifies the DFT parameters in the generated input file upon repeated failures.

*   `PhysicsInformedGenerator` (`generator.py`): This class acts as a factory for `Atoms` objects. It will contain methods like `generate_sqs`, `apply_strain`, `add_rattling`, etc. It will be initialized with the relevant section of the `SystemConfig` and will use external libraries like `icet` and `pymatgen` to perform the generation tasks.

*   `DatabaseManager` (`database.py`): This will be a thin wrapper around the ASE `connect` function. Its primary role is to extend the standard database functionality by providing methods to write and read the custom metadata specified in the `ALL_SPEC.md`, such as `config_type`, `uncertainty_gamma`, and the `force_mask` array. This ensures all crucial information is stored alongside the atomic structures.

The overall design promotes loose coupling. Modules do not call each other directly; instead, they communicate through well-defined data structures (the database and queues). This makes the system easier to test, maintain, and extend. For instance, replacing Quantum Espresso with a different DFT code would only require reimplementing the `QEProcessRunner`, leaving the rest of the system untouched.

---

## 5. Implementation Plan

The development of MLIP-AutoPipe is structured into six sequential cycles. Each cycle delivers a concrete, testable piece of the final system, building upon the foundation of the previous ones. This iterative approach allows for continuous integration and reduces risk.

**CYCLE 01: The Foundation - Schemas and DFT Factory Core**
*   **Objective:** To establish the project's backbone, including the data structures that will govern the entire application and a basic, workable interface to the DFT engine.
*   **Features:**
    1.  **Project Scaffolding:** Initialise the repository with the file structure outlined in the Design Architecture. Configure `pyproject.toml` with all project dependencies (`pydantic`, `ase`, `typer`, etc.) and set up the `uv` virtual environment.
    2.  **Pydantic Schema Definition:** Implement the complete set of Pydantic models in `src/mlip_autopipec/config/`. This includes `UserConfig` and the comprehensive `SystemConfig`. The focus is on defining the structure and validation rules for all configuration parameters that will be used throughout the project.
    3.  **Core DFT Runner:** Implement a first version of the `QEProcessRunner` class in `dft_factory.py`. This version will be able to take an ASE `Atoms` object and a `SystemConfig`, generate a valid Quantum Espresso input file, execute `pw.x` as a subprocess, and parse the output to extract energy and forces.
    4.  **Database Schema:** Implement the `DatabaseManager` in `data/database.py`. This includes setting up the connection to an SQLite file and adding the custom metadata columns (`config_type`, `uncertainty_gamma`, `force_mask`) to the database schema.
*   **Outcome:** A developer can programmatically define a calculation using the Pydantic models, execute a single DFT calculation, and see the results correctly stored in the database with the appropriate metadata.

**CYCLE 02: The Seed - Physics-Informed Generator**
*   **Objective:** To build the module responsible for creating the initial, diverse set of structures, solving the "cold-start" problem.
*   **Features:**
    1.  **Generator Class:** Implement the `PhysicsInformedGenerator` class in `generator.py`.
    2.  **Alloy Generation:** Integrate the `icet` library to implement SQS generation. Add methods to apply volumetric and shear strain to the generated SQS cells. Implement the "rattling" functionality with configurable standard deviations.
    3.  **Crystal Defect Generation:** Integrate the `pymatgen` library to systematically introduce point defects (vacancies, interstitials, antisites) into conventional crystal structures.
    4.  **Workflow Integration:** Create a simple script or CLI command that uses the `PhysicsInformedGenerator` to create a set of structures and saves them as a list of `Atoms` objects or directly into the project's database.
*   **Outcome:** The system can generate thousands of diverse atomic structures for alloys and crystals based on the `UserConfig`, providing the raw material for the first training cycle.

**CYCLE 03: The Filter - Surrogate Explorer and Selector**
*   **Objective:** To implement the intelligent data selection mechanism that drastically reduces the number of required DFT calculations.
*   **Features:**
    1.  **Surrogate Integration:** Implement the `SurrogateExplorer` class in `explorer.py`. Integrate the `mace-torch` library to load a pre-trained MACE model. Create a method to perform rapid energy and force prediction on a list of `Atoms` objects.
    2.  **Descriptor Calculation:** Add functionality to compute structural descriptors for each `Atoms` object. The initial implementation will use the SOAP descriptor, available through the `dscribe` library.
    3.  **Farthest Point Sampling (FPS):** Implement the FPS algorithm. This function will take the full list of descriptors and select a smaller, maximally diverse subset based on their Euclidean distance in descriptor space.
    4.  **Pipeline Connection:** Create a workflow that chains the modules: The output from the Generator (Cycle 02) is fed into the Surrogate Explorer, which first filters out high-energy structures and then uses FPS to select the final candidates to be sent to the DFT queue.
*   **Outcome:** A complete "cold-start" data selection pipeline exists. Given a user config, the system generates thousands of structures, filters them down to a few hundred information-rich candidates, and prepares them for submission to the DFT Factory.

**CYCLE 04: The Factory - Pacemaker Trainer and Database Loop**
*   **Objective:** To close the first loop: training an actual MLIP from the data generated and stored in the database.
*   **Features:**
    1.  **Trainer Class:** Implement the `PacemakerTrainer` class in `trainer.py`.
    2.  **Input Automation:** Write code to automatically generate the YAML input file required by Pacemaker for training, populating it with parameters from the `SystemConfig`.
    3.  **Data Acquisition:** The trainer will query the `DatabaseManager` to retrieve all successfully completed DFT calculations.
    4.  **Training Execution:** The class will wrap the `pacemaker_train` command-line tool, executing it as a subprocess and monitoring its success.
*   **Outcome:** The system can take the DFT-calculated structures from the database, automatically configure and launch a Pacemaker training job, and produce a version 1 MLIP file.

**CYCLE 05: The Intelligence - Full Active Learning Loop**
*   **Objective:** To connect all modules and implement a complete, functioning active learning cycle.
*   **Features:**
    1.  **Inference Engine:** Implement a `LammpsRunner` class in `inference.py` to execute MD simulations using the latest trained MLIP.
    2.  **Uncertainty Quantification:** During the MD run, leverage Pacemaker's functionality to calculate the extrapolation grade for each step.
    3.  **Feedback Trigger:** When the uncertainty exceeds a threshold defined in `SystemConfig`, the MD simulation will pause.
    4.  **Loop Closure:** The high-uncertainty structure will be extracted and placed into the DFT queue. This will trigger the DFT Factory (Cycle 01) and, subsequently, the Pacemaker Trainer (Cycle 04) to refine the model.
    5.  **Main Application:** Develop the main CLI application in `app.py` using Typer, which orchestrates this entire loop.
*   **Outcome:** A functioning, end-to-end prototype of the MLIP-AutoPipe. It can start with a basic potential, improve it through active learning, and demonstrate a tangible increase in the model's robustness.

**CYCLE 06: Production Readiness - Advanced Embedding and Scalability**
*   **Objective:** To implement the most advanced features of the system and ensure it is robust and scalable for production use.
*   **Features:**
    1.  **Periodic Embedding:** Implement the advanced periodic embedding logic in `inference.py`. When an uncertain atom is identified, this code will extract a small periodic sub-cell around it, rather than just the single structure.
    2.  **Force Masking:** Implement the corresponding force masking. When a periodically embedded cell is added to the database, a `force_mask` array will be generated and stored, indicating that forces on buffer atoms should be ignored during training.
    3.  **Robust Error Handling:** Enhance the `QEProcessRunner` with the full, multi-level auto-recovery logic (e.g., adjusting mixing_beta, degauss, diagonalization algorithm).
    4.  **Task Queue Integration:** Replace the simple subprocess calls with a proper task queue framework (e.g., Dask). This will allow for the parallel execution of hundreds or thousands of DFT calculations on an HPC cluster, transforming the prototype into a high-throughput system.
    5.  **Checkpointing:** Implement robust checkpointing so that a long-running workflow can be safely stopped and restarted.
*   **Outcome:** A production-ready, scalable, and highly efficient autonomous MLIP generation pipeline that leverages state-of-the-art techniques for data generation and active learning.

---

## 6. Test Strategy

A rigorous, multi-layered testing strategy is essential to ensure the reliability and correctness of the MLIP-AutoPipe system. The strategy combines unit tests for individual components, integration tests for module interactions, and end-to-end tests for the entire workflow. Testing will be a core part of each development cycle.

**General Principles:**

*   **Test-Driven Development (TDD):** For complex algorithmic components like the FPS selector or the auto-recovery logic, a TDD approach will be encouraged.
*   **Mocking and Patching:** External dependencies, especially computationally expensive ones like Quantum Espresso and LAMMPS, will be heavily mocked. The goal is to test the system's logic (e.g., "Did it generate the correct input file and command?") rather than the external tools themselves. The `pytest-mock` library will be used extensively.
*   **Continuous Integration (CI):** A CI pipeline will be set up to automatically run the entire test suite on every commit, ensuring that new changes do not introduce regressions.
*   **Code Coverage:** The project will aim for a minimum of 85% test coverage for all new code, enforced by the CI pipeline.

**Testing by Cycle:**

*   **CYCLE 01 (Foundation & DFT Factory):**
    *   **Unit Tests:** Test the Pydantic schemas: ensure that valid configurations are accepted and invalid ones raise the correct `ValidationError`. Test edge cases for parameter validation. For the `QEProcessRunner`, unit tests will focus on the input file generation logic. Given a specific `SystemConfig` and `Atoms` object, does the generated QE input string contain the exact keywords and values expected?
    *   **Integration Tests:** The key integration test will be for the `QEProcessRunner`'s interaction with the file system and subprocesses. Using mocks for `subprocess.run`, we will test the full `run_calculation` method. This includes mocking a successful run, a failed run (e.g., non-zero exit code), and parsing of a sample QE output file. We will verify that the correct data is written to the mocked database.

*   **CYCLE 02 (Generator):**
    *   **Unit Tests:** Test each generation method (`generate_sqs`, `apply_strain`) in isolation. For instance, verify that `apply_strain` correctly modifies the lattice parameters of an `Atoms` object.
    *   **Integration Tests:** Test the main `PhysicsInformedGenerator` class. Provide a `SystemConfig` for a specific material type (e.g., alloy) and verify that the `run()` method produces the expected number and type of structures. Check that the output structures are physically reasonable (e.g., no overlapping atoms).

*   **CYCLE 03 (Explorer & Selector):**
    *   **Unit Tests:** Unit test the FPS algorithm with a known set of 2D points to ensure it selects the correct sequence of farthest points. Unit test the descriptor calculation wrappers.
    *   **Integration Tests:** Create a test fixture with a list of simple `Atoms` objects. Test the `SurrogateExplorer`'s main method: does it correctly call the (mocked) MACE model? Does it filter the structures based on the mocked energy output? Does the final output list contain the expected number of structures after FPS selection?

*   **CYCLE 04 (Trainer):**
    *   **Unit Tests:** Test the Pacemaker input file generation. Given a `SystemConfig`, does it produce the correct YAML string?
    *   **Integration Tests:** Mock the `DatabaseManager` to return a predefined set of training data. Mock the `subprocess.run` call to the `pacemaker_train` command. The test will verify that the `PacemakerTrainer` class correctly queries the database, generates the right config, calls the training command with the correct arguments, and identifies the resulting potential file.

*   **CYCLE 05 (Active Learning Loop):**
    *   **Integration Tests:** This cycle requires the most complex integration tests. We will mock the `LammpsRunner` to simulate an MD run that produces a high uncertainty at a specific timestep. We will then verify that this triggers the entire chain of events: the structure is sent to the DFT queue, a mocked `QEProcessRunner` "calculates" it, the result is placed in the database, and a mocked `PacemakerTrainer` "retrains" the model. This tests the correct orchestration of modules by the main application.

*   **CYCLE 06 (Advanced Features):**
    *   **Unit Tests:** Write specific unit tests for the periodic embedding algorithm. Given a large `Atoms` object and an atomic index, does the function correctly extract the smaller periodic cell with the right atoms and lattice vectors? Write unit tests for the force mask generation, ensuring buffer atoms are correctly assigned a weight of zero.
    *   **Integration Tests:** Test the enhanced `QEProcessRunner`'s auto-recovery logic. Mock `subprocess.run` to fail multiple times, and assert that the runner correctly modifies the generated input file with the expected fallback parameters (e.g., `mixing_beta` is reduced) on each subsequent attempt.
