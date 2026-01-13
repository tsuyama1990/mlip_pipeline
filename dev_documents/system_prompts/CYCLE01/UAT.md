# CYCLE01 User Acceptance Testing: The Foundation

This document outlines the User Acceptance Testing (UAT) plan for CYCLE01. The primary goal of this UAT is to allow a user, such as a materials scientist, to verify that the foundational data generation pipeline of the MLIP-AutoPipe system is functioning correctly and producing physically meaningful results. This cycle's success is defined by its ability to autonomously generate a high-quality dataset that can serve as the basis for training a reliable machine learning potential. The UAT is designed to be a hands-on, intuitive experience, using Jupyter notebooks to guide the user through the process and to visualize the results in a clear and understandable way.

## 1. Test Scenarios

The UAT for CYCLE01 is centered around a single, comprehensive test scenario that validates the core functionality of both the Physics-Informed Generator (Module A) and the Automated DFT Factory (Module C). The chosen system is elemental Silicon (Si) in the diamond structure, a cornerstone material in semiconductor physics and a well-understood benchmark system.

| Scenario ID | Description | Priority |
| :--- | :--- | :--- |
| **UAT-C1-001** | **Generate and Validate Silicon Equation of State (EOS) Dataset** | **High** |

**Scenario UAT-C1-001 Details:**

*   **Objective**: To verify that the system can correctly generate a series of strained Silicon crystal structures and calculate their corresponding energies using the automated DFT factory, ultimately producing a physically correct Equation of State (EOS) curve. The EOS curve, which shows how the energy of a crystal changes with its volume, is a fundamental property and a critical test of the system's ability to handle solid-state calculations.
*   **User Story**: "As a materials scientist, I want to provide the system with the chemical symbol for Silicon and its crystal structure. I expect the system to automatically generate a set of uniformly strained supercells, run high-quality DFT calculations on them without any manual intervention, and present me with a database of the results. I want to be able to easily plot the energy-volume curve from this database to confirm that it is physically correct and that the system has found the correct equilibrium lattice constant."
*   **Methodology**: The user will be provided with a single Jupyter Notebook (`UAT_CYCLE01.ipynb`). This notebook will guide them through the following steps:
    1.  **Configuration**: Defining a simple input configuration for the test case in a Python dictionary, which will then be written to a YAML file. This mimics the intended user experience.
    2.  **Execution**: Running the main CYCLE01 command-line script from within the notebook. The notebook will execute the entire data generation pipeline, showing the user the real-time output of the system as it generates structures and runs DFT calculations.
    3.  **Verification and Visualization**: Once the pipeline is complete, the notebook will contain cells to load the generated ASE database file. It will then use the data to perform the following checks:
        *   Confirm that the expected number of structures were created and calculated.
        *   Extract the lattice constant and total energy for each structure.
        *   Plot the Energy vs. Volume curve.
        *   Fit the curve to a standard equation of state (e.g., the Birch-Murnaghan EOS) to extract the equilibrium lattice constant, bulk modulus, and minimum energy.
        *   Compare these results against well-established experimental or literature values for Silicon.
*   **Why this UAT is amazing for the user**: This UAT provides a transparent and educational window into the core of the MLIP-AutoPipe system. Instead of just seeing a final result, the user actively participates in the configuration, execution, and validation steps in a simple, interactive environment. The immediate visual feedback of plotting a correct EOS curve provides a powerful and satisfying confirmation that the complex underlying machinery of automated DFT is working as it should. It demystifies the "black box" and builds trust in the system's capabilities from the very first cycle.

## 2. Behavior Definitions

The following Gherkin-style definitions describe the expected behavior of the system for the UAT scenario.

### Scenario: Generate and Validate Silicon Equation of State (EOS) Dataset

*   **GIVEN** a user who wants to generate a training dataset for elemental Silicon.
*   **AND** they have access to the Jupyter Notebook `UAT_CYCLE01.ipynb`.

*   **WHEN** they define a configuration specifying the element as "Si" and the crystal structure as "diamond".
*   **AND** they specify a generation goal of "eos_strain" with a strain range of -5% to +5%.
*   **AND** they execute the main pipeline command from within the notebook.

*   **THEN** the system should start the automated data generation process without requiring any further input.
*   **AND** Module A (the Generator) should create a series of Silicon diamond-structure supercells, each with a different, uniform volumetric strain applied.
*   **AND** Module C (the DFT Factory) should receive each of these structures.
*   **AND** the DFT Factory's heuristic engine should automatically select the correct SSSP pseudopotential and a suitable plane-wave cutoff for Silicon.
*   **AND** it should automatically determine an appropriate k-point mesh for the supercell size.
*   **AND** it should successfully execute a Quantum Espresso calculation for each of the strained structures, handling any potential convergence issues without user intervention.
*   **AND** a database file (e.g., `Si_eos.db`) should be created containing the results.
*   **AND** this database must contain the final energy, forces, and stress for each structure.

*   **GIVEN** the successful completion of the data generation pipeline.

*   **WHEN** the user executes the verification cells in the Jupyter Notebook.
*   **AND** the notebook reads the `Si_eos.db` file.

*   **THEN** the notebook should display a plot of Energy versus Volume for the calculated structures.
*   **AND** the plot should show a smooth, parabolic curve with a clear minimum.
*   **AND** a Birch-Murnaghan fit to the data should yield an equilibrium lattice constant for Silicon that is within 1% of the accepted literature value (approx. 5.43 Å).
*   **AND** the calculated bulk modulus should be within 10% of the accepted literature value (approx. 98 GPa).
*   **AND** the user should be able to see a confirmation message in the notebook stating that the UAT has passed successfully.
