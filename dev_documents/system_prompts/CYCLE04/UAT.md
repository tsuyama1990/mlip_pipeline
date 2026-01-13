# CYCLE04 User Acceptance Testing: The User Experience

This document outlines the User Acceptance Testing (UAT) plan for CYCLE04. The focus of this cycle is a significant shift towards user-friendliness and accessibility. The UAT is designed to validate that the complex, autonomous system built in previous cycles can now be easily and intuitively controlled by a user. The user will experience the full, polished workflow: starting a complex simulation with a remarkably simple input file, using a professional command-line tool, and monitoring the live progress of the autonomous learning process through a clean, web-based interface. This UAT aims to confirm that the system is not just powerful, but also a pleasure to use.

## 1. Test Scenarios

The UAT for CYCLE04 is structured as a single, seamless user journey that tests all the new user-facing components in a logical workflow.

| Scenario ID | Description | Priority |
| :--- | :--- | :--- |
| **UAT-C4-001** | **Launch and Monitor a "Zero-Input" Simulation of Nickel** | **High** |

**Scenario UAT-C4-001 Details:**

*   **Objective**: To allow a user to experience the full, simplified workflow for a common materials science task: running a simulation to equilibrate a crystal and calculate its lattice constant. The user will verify the ease of configuration, the professionalism of the CLI, and the clarity of the monitoring dashboard.
*   **User Story**: "As a materials scientist, I don't want to spend my time tweaking hundreds of low-level parameters. I want to tell the system what material I'm interested in (e.g., Nickel) and my high-level goal (e.g., 'find the equilibrium structure'). I then expect the system to figure out all the details. I want to launch this job from a proper command-line tool, and then I want to be able to easily check on its progress from a web browser to see if it's learning correctly and to know when it's finished."
*   **Methodology**: The UAT will be conducted using a Jupyter Notebook (`UAT_CYCLE04.ipynb`) to guide the user, but the core interactions will be with the actual CLI and the web browser.
    1.  **Act 1 - The Minimalist Configuration**: The notebook will first instruct the user to create a very simple YAML file, `nickel.yaml`. This file will be remarkably short, for example:
        ```yaml
        project_name: "UAT_Nickel_Equilibrium"
        target_system:
          elements: ["Ni"]
          crystal_structure: "fcc"
        simulation_goal: "equilibrate"
        ```
        The user will appreciate that they do not need to specify any DFT parameters, simulation temperatures, or other complex settings.
    2.  **Act 2 - The Professional CLI**: The user will then move to a terminal within the notebook environment. They will be guided to test the new CLI's features:
        *   They will run `mlip-auto --help` to see the well-structured help message.
        *   They will run `mlip-auto run --help` to see the specific options for the `run` command.
        *   They will then launch the main process by running `mlip-auto run --input nickel.yaml`. The system will start, and the user will see clean, informative log messages in their terminal.
    3.  **Act 3 - The Live Dashboard**: While the backend is running, the user will be instructed to run `mlip-auto dashboard` in a separate terminal. This will launch the web server.
        *   The user will open a web browser to the provided URL (e.g., `http://127.0.0.1:8000`).
        *   They will be presented with the live monitoring dashboard. They will be asked to observe the different charts. They should see the training RMSE decrease over time (as the model learns) and the number of structures in the DFT queue fluctuate as the OTF loop identifies and processes new configurations.
        *   The simulation for the UAT will be configured to be short, so the user can see it through to completion.
    4.  **Act 4 - The Verification**: After the run is complete, the user will be guided to check the results. The system will have produced a report or a final dataset. The notebook will help the user load this result and confirm that the system correctly determined the equilibrium lattice constant for Nickel, comparing it to the known literature value.
*   **Why this UAT is amazing for the user**: This UAT directly addresses the primary pain points of traditional computational science: complexity and opacity. The user experiences the magic of the Heuristic Engine by providing a tiny input file and having the system make all the right expert-level decisions. The professional CLI makes them feel they are using a robust piece of software, not a messy script. The live dashboard transforms the "black box" of the simulation into a transparent process they can watch and understand. It's a deeply satisfying workflow that builds confidence and trust in the system's autonomy.

## 2. Behavior Definitions

The following Gherkin-style definitions describe the expected behavior of the system for the UAT scenario.

### Scenario: Launch and Monitor a "Zero-Input" Simulation of Nickel

*   **GIVEN** a user with access to the MLIP-AutoPipe command-line tool.
*   **AND** they have created a minimal `nickel.yaml` file specifying "Ni", "fcc" structure, and the goal "equilibrate".

*   **WHEN** they run the command `mlip-auto run --input nickel.yaml` in their terminal.
*   **THEN** the system should start the full OTF active learning pipeline without any further prompts or errors.
*   **AND** the Heuristic Engine must automatically select appropriate DFT parameters for Nickel, including enabling spin polarization (`nspin=2`).
*   **AND** it must set up a simulation designed to find the equilibrium structure, which may involve running simulations at various cell volumes.
*   **AND** the system should begin logging key metrics to a `metrics.jsonl` file in its output directory.

*   **GIVEN** the main `run` process is active in the background.

*   **WHEN** the user runs the command `mlip-auto dashboard` in a new terminal.
*   **AND** they navigate to the specified URL in their web browser.
*   **THEN** a web page must load with a title like "MLIP-AutoPipe Dashboard".
*   **AND** the dashboard must display several charts, including a chart for "Training RMSE" and "DFT Queue Size".
*   **AND** these charts must update automatically without the user needing to refresh the page.
*-   **AND** the user should be able to observe the RMSE value decrease after the system performs a retraining step.

*   **GIVEN** the `run` process has completed successfully.

*   **WHEN** the user inspects the output directory.
*   **THEN** there must be a final, trained potential file (`.yace`).
*   **AND** there must be an output file or report summarizing the results.
*   **AND** the reported equilibrium lattice constant for Nickel must be within 1% of the accepted experimental value (approx. 3.52 Å).
*   **AND** the user should receive a confirmation that the process is complete and the UAT has been passed.
