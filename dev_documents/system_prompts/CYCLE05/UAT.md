# CYCLE05 User Acceptance Testing: The Power-Up

This document outlines the User Acceptance Testing (UAT) plan for CYCLE05. The focus of this cycle is to validate the major expansion in the system's scientific capabilities. The UAT is designed to allow a user to go beyond just building a potential and to use the system to answer a complex, scientifically relevant question about a material's properties. The user will experience how the Heuristic Engine can now plan and execute a multi-stage workflow to calculate an advanced property, demonstrating that MLIP-AutoPipe has evolved into a true automated materials science platform.

## 1. Test Scenarios

The UAT for CYCLE05 will be a single, in-depth scenario that guides the user through one of the new, advanced workflows. We will focus on the calculation of elastic constants, as this is a fundamental mechanical property with a clear, verifiable outcome.

| Scenario ID | Description | Priority |
| :--- | :--- | :--- |
| **UAT-C5-001** | **Automatically Calculate the Elastic Constants of Aluminum** | **High** |

**Scenario UAT-C5-001 Details:**

*   **Objective**: To allow a user to verify that the system can successfully plan and execute the complete, multi-step workflow required to calculate the full elastic tensor ($C_{ij}$) of a material. This test will validate the Heuristic Engine's new "expert" knowledge and the new analysis modules.
*   **User Story**: "As a materials engineer, I need to know the mechanical properties of a new alloy I'm designing. Calculating the full set of elastic constants is a tedious, multi-step process that is easy to mess up. I want to simply tell the system the material I'm working with (e.g., Aluminum) and ask it to calculate the elastic constants. I expect it to automatically perform the necessary structure relaxation, apply the correct set of deformations, run the DFT calculations, and then present me with the final, validated elastic tensor."
*   **Methodology**: The UAT will be conducted through a Jupyter Notebook (`UAT_CYCLE05.ipynb`), which will provide guidance and visualization for a workflow run using the command-line tool.
    1.  **Act 1 - The Simple Request**: The user will begin by creating a minimal `aluminum.yaml` input file. The key change will be the `simulation_goal`:
        ```yaml
        project_name: "UAT_Aluminum_Elastic"
        target_system:
          elements: ["Al"]
          crystal_structure: "fcc"
        simulation_goal: "elastic"
        ```
        The user will appreciate that this simple, high-level request is all that's needed to trigger a complex, multi-stage calculation.
    2.  **Act 2 - The Orchestrated Execution**: The user will launch the run via the `mlip-auto run` command. The notebook will explain what the system is doing in the background, building a mental model for the user:
        *   "First, the system is performing a full geometry optimization to find the precise equilibrium lattice constant of Aluminum at 0K. This uses the active learning loop to ensure the potential is highly accurate around the equilibrium state."
        *   "Next, the system is generating a specific set of 6-12 strained versions of the optimized crystal. These are not random strains; they are the precise deformations needed to derive the elastic constants."
        *   "Now, the system is dispatching these structures to the DFT queue for calculation. You can monitor the progress on the dashboard." (The user will be prompted to open the dashboard and see the batch of DFT jobs being processed).
    3.  **Act 3 - The Analysis and Result**: Once all the DFT calculations are finished, the system will automatically run the final analysis step.
        *   The notebook will inform the user that the run is complete and that the results have been generated.
        *   The user will execute a cell in the notebook that loads the final results file (e.g., `results.json`).
        *   The notebook will display the calculated elastic constants, $C_{11}, C_{12}, \text{and } C_{44}$, in a clean, tabular format.
        *   Crucially, the notebook will then compare these calculated values against established literature values for Aluminum, showing the percentage error. It will also verify that the calculated constants obey the rules for cubic crystal stability (e.g., $C_{11} > C_{12}$, $C_{44} > 0$).
*   **Why this UAT is amazing for the user**: This UAT demonstrates a significant leap in the system's utility. The user isn't just building a tool (a potential); they are using the system to get a direct, actionable scientific answer. The workflow demystifies a complex and error-prone research task, showing how automation can encapsulate expert knowledge to make advanced calculations accessible and reliable. Seeing the final, accurate elastic constants appear after such a simple initial request provides a powerful sense of the system's leverage and potential for accelerating their own research.

## 2. Behavior Definitions

The following Gherkin-style definitions describe the expected behavior of the system for the UAT scenario.

### Scenario: Automatically Calculate the Elastic Constants of Aluminum

*   **GIVEN** a user with access to the MLIP-AutoPipe system.
*   **AND** they have created a minimal `aluminum.yaml` file with the `simulation_goal` set to `"elastic"`.

*   **WHEN** they execute the command `mlip-auto run --input aluminum.yaml`.
*   **THEN** the system must start the multi-stage elastic constant workflow without any further user input.
*   **AND** the Heuristic Engine must correctly interpret the `"elastic"` goal.
*   **AND** the system must first perform a full cell relaxation to find the equilibrium structure.
*   **AND** the system must then generate a small, specific set of strained structures based on the relaxed cell.
*   **AND** a series of DFT calculation jobs for these structures must be dispatched to the task queue.

*   **GIVEN** the workflow is running.

*   **WHEN** the user opens the monitoring dashboard.
*   **THEN** they should be able to see a batch of DFT calculations being processed by the Celery workers.

*   **GIVEN** all DFT calculations for the strained structures have been successfully completed.

*   **WHEN** the system enters the final analysis phase.
*   **THEN** the `fit_elastic_constants` analysis function must be executed automatically.
*   **AND** a final results file must be created in the output directory.
*   **AND** this file must contain the calculated values for the elastic constants $C_{11}, C_{12}, \text{and } C_{44}$.

*   **GIVEN** the results have been generated.

*   **WHEN** the user runs the verification script in the `UAT_CYCLE05.ipynb` notebook.
*   **THEN** the notebook must load and display the calculated elastic constants.
*   **AND** the calculated values for $C_{11}, C_{12}, \text{and } C_{44}$ for Aluminum must be within 5% of their accepted literature values (approx. 107 GPa, 61 GPa, and 28 GPa, respectively).
*   **AND** a final confirmation message must be displayed, confirming that the UAT was successful.
