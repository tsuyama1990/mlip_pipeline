# CYCLE02 User Acceptance Testing: The Apprentice

This document outlines the User Acceptance Testing (UAT) plan for CYCLE02. The focus of this cycle is to validate the newly introduced "intelligence" and efficiency in our pipeline. Users will verify two key advancements: the ability of the **Surrogate Explorer** to drastically reduce the required DFT calculations through intelligent selection, and the capability of the **Trainer Module** to produce a valid, working Machine Learning Interatomic Potential (MLIP) from the curated data. This UAT is designed to give the user a tangible sense of the power of machine learning in this context, demonstrating how a small, smart dataset can be used to create a model that accurately captures complex physical behaviour.

## 1. Test Scenarios

The UAT for CYCLE02 builds directly on the Silicon benchmark from the previous cycle. It is broken into two scenarios to test the data selection and training processes independently, before combining them.

| Scenario ID | Description | Priority |
| :--- | :--- | :--- |
| **UAT-C2-001** | **Verify Intelligent Data Selection with the Surrogate Explorer** | **High** |
| **UAT-C2-002** | **Train and Validate a Silicon MLIP** | **High** |

**Scenario UAT-C2-001 Details:**

*   **Objective**: To allow the user to visualize and confirm that the Surrogate Explorer (Module B) is selecting a structurally diverse subset of candidate structures from a much larger initial set. This demonstrates the core value proposition of CYCLE02: saving massive computational cost.
*   **User Story**: "As a materials scientist, I'm skeptical that a small subset of structures can represent a large, complex dataset. I want to generate a large number of rattled and strained silicon structures and then use the new Surrogate Explorer to select a small fraction of them. I need to be able to visualize the structural diversity of both the initial and the selected sets to convince myself that the selection process is intelligent and not just random."
*   **Methodology**: A Jupyter Notebook (`UAT_CYCLE02.ipynb`) will be the user's interface.
    1.  **Generation**: The user will first execute a step that runs only Module A (Generator) to produce a large set of ~1000 silicon structures with various strains and random rattles.
    2.  **Visualization of Initial Set**: To visualize the "structural diversity", we will use a dimensionality reduction technique like PCA (Principal Component Analysis) on the SOAP descriptors of the structures. The notebook will plot the first two principal components of all 1000 structures, showing them as a large cloud of points.
    3.  **Execution of Explorer**: The user will then run Module B (Surrogate Explorer) on this large set, configured to select just 50 structures using Farthest Point Sampling (FPS).
    4.  **Visualization of Selected Set**: The notebook will then create the same PCA plot but will highlight the 50 selected points. The user will be able to visually confirm that the highlighted points are spread out across the entire cloud, covering the extremities and the center, rather than being clumped in one region. This provides intuitive proof of the diversity of the selection.

**Scenario UAT-C2-002 Details:**

*   **Objective**: To guide the user through the process of training their first MLIP and then immediately using it to predict a fundamental physical property, the Equation of State (EOS), validating its accuracy against the ground-truth DFT data.
*   **User Story**: "Now that I have a small, high-quality dataset, I want to train my own potential. I need a simple way to start the training process. Once it's done, I want to immediately test the new potential by using it to predict the energy-volume curve of silicon, and I want to see a direct comparison of my MLIP's prediction against the original DFT data to check if it's accurate."
*   **Methodology**: This scenario continues in the same Jupyter Notebook (`UAT_CYCLE02.ipynb`).
    1.  **Data Generation**: The user will run the fully integrated `generate` command, which executes Modules A, B, and C in sequence. This will generate 1000 structures, select 50 via FPS, and run DFT calculations on them, saving the results to a database (`Si_fps.db`).
    2.  **Training**: The user will execute a notebook cell that calls the new `mlip-auto train` command, pointing to the `Si_fps.db` database. The notebook will display the training progress from Pacemaker's output.
    3.  **Validation**: Once training is complete, a new potential file (`.yace`) is created. The notebook will:
        *   Load this new MLIP as an ASE calculator.
        *   Use it to calculate the energies for a range of strained silicon structures (the same ones used for the DFT EOS curve). This is a fast, pure-inference step.
        *   Load the original DFT results for the EOS curve (from the CYCLE01 UAT or a cached file).
        *   Plot both the DFT energy-volume points and the continuous curve predicted by the new MLIP on the same graph.
    *   **Why this UAT is amazing for the user**: This scenario delivers the "magic moment" of the project so far. The user witnesses the transformation of raw DFT data into a lightweight, fast, and accurate predictive model. The final plot, showing their own trained MLIP's curve passing perfectly through the expensive DFT data points, provides a powerful and immediate validation of the entire process. It's a direct, visual confirmation of the potential's predictive power.

## 2. Behavior Definitions

The following Gherkin-style definitions describe the expected behavior of the system for the UAT scenarios.

### Scenario: Verify Intelligent Data Selection with the Surrogate Explorer

*   **GIVEN** a user with the `UAT_CYCLE02.ipynb` notebook.
*   **AND** they have configured the system to generate 1000 structures for Silicon with various strains and rattles.

*   **WHEN** they execute the generation step (Module A).
*   **AND** they execute the visualization step for the initial dataset.
*   **THEN** a 2D plot should be displayed showing a large cloud of approximately 1000 points, representing the descriptors of the initial structures.

*   **WHEN** they execute the explorer step (Module B) configured to select 50 structures.
*   **AND** they execute the visualization step for the selected dataset.
*   **THEN** the same 2D plot should be displayed, but with 50 of the points highlighted.
*   **AND** the user must be able to visually confirm that the highlighted points are distributed widely across the cloud, not clustered together, demonstrating the diversity of the FPS selection.

### Scenario: Train and Validate a Silicon MLIP

*   **GIVEN** a user with the `UAT_CYCLE02.ipynb` notebook.
*   **AND** they have successfully run the full data generation and selection pipeline (`run` command) to produce a database (`Si_fps.db`) of 50 DFT-calculated structures.

*   **WHEN** they execute the training step, pointing the `train` command to the `Si_fps.db` database.
*   **THEN** the system should start the Pacemaker training process without any further input.
*   **AND** the notebook should display the output log from the training process, showing the decreasing error (RMSE).
*   **AND** upon completion, a new potential file (e.g., `Si_potential.yace`) must be created in the output directory.

*   **GIVEN** the successful creation of the `Si_potential.yace` file.

*   **WHEN** the user executes the validation cells in the notebook.
*   **THEN** the notebook should load the trained potential as an ASE calculator.
*   **AND** it should generate a plot containing two sets of data:
    1.  A scatter plot of the original DFT energy vs. volume points.
    2.  A smooth line plot of the MLIP-predicted energy vs. volume.
*   **AND** the user must be able to visually confirm that the MLIP's curve passes through or very close to the DFT data points.
*   **AND** the notebook should calculate and display the Root Mean Squared Error (RMSE) in energy between the MLIP prediction and the DFT data, which should be below a defined low threshold (e.g., 5 meV/atom).
*   **AND** a confirmation message should be displayed, stating that the UAT has passed.
