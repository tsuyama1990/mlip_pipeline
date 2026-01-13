# CYCLE06 User Acceptance Testing: The Final Polish

This document outlines the User Acceptance Testing (UAT) plan for CYCLE06. This is the final UAT, and its goal is to validate the complete user journey from discovering the project to successfully installing and running it. The focus is on the "out-of-box" experience. The user will test the clarity of the documentation, the simplicity of the installation process, and the robustness of the final packaged application. This UAT will confirm that MLIP-AutoPipe is not just a powerful tool, but also a professional, accessible, and well-supported piece of software ready for the scientific community.

## 1. Test Scenarios

The UAT for this final cycle is broken into three scenarios, each testing a critical part of the new user's journey: documentation, installation, and first contact with the software.

| Scenario ID | Description | Priority |
| :--- | :--- | :--- |
| **UAT-C6-001** | **Can a New User Understand the Project from its Documentation?** | **High** |
| **UAT-C6-002** | **Can a New User Easily Install the Software?** | **High** |
| **UAT-C6-003** | **Does the Packaged Application Run Correctly?** | **High** |

**Scenario UAT-C6-001 Details:**

*   **Objective**: To ensure that a user who is completely new to the project can understand its purpose, how to install it, and how to use it, based solely on the official documentation website.
*   **User Story**: "As a researcher looking for a tool to automate my potential fitting, I've just found the MLIP-AutoPipe documentation website. I need to be able to quickly understand what this software does, if it's right for me, and how to get started, without having to ask for help or read the source code."
*   **Methodology**: The user will be given a single URL to the (locally hosted or pre-built) documentation website. They will be asked to perform the following tasks and answer questions:
    1.  **Exploration**: Read the home page (`index.md`). **Question**: "In your own words, what problem does MLIP-AutoPipe solve?"
    2.  **Installation Guide**: Navigate to the "Installation" page. **Question**: "Which installation method seems best for you, and are the instructions clear enough for you to follow?"
    3.  **Tutorial**: Navigate to the "Usage" section and find the tutorial for calculating elastic constants. Read through the tutorial. **Question**: "Does this tutorial give you a clear idea of how you would use the software for a real research task?"
    4.  **API Reference**: Find the "Developer / API Reference" section. **Question**: "If you wanted to understand the exact structure of the `UserConfig` input file, could you find that information here easily?"

**Scenario UAT-C6-002 Details:**

*   **Objective**: To validate that the installation process is smooth, simple, and works as described in the documentation.
*   **User Story**: "I've decided to try MLIP-AutoPipe. I want to install it on my local machine with a single, simple command, as promised in the documentation. I expect this process to complete without errors and without me needing to manually install a long list of complicated dependencies."
*   **Methodology**: The user will be provided with a clean, standard Python virtual environment.
    1.  **PyPI Installation**: The user will follow the documentation's instructions to install the package from a local directory (simulating PyPI) using `pip`. The command should be as simple as `pip install /path/to/mlip_autopipec/dist/package.whl`.
    2.  **Verification**: After the installation command completes, the user will be asked to confirm that it finished without any errors.
    3.  **Container Installation (Optional)**: If the user is familiar with Docker, they will be guided to build the Docker image using the provided `Dockerfile` and confirm that the build process completes successfully.

**Scenario UAT-C6-003 Details:**

*   **Objective**: To give the user a "first contact" experience with the installed software and confirm that the packaging and entry points are configured correctly.
*   **User Story**: "Now that the software is installed, I want to make sure it's working. I expect a command-line tool `mlip-auto` to be available in my path. I want to run a simple command to see its help message and confirm that the program launches correctly."
*   **Methodology**: Continuing from the previous scenario in the same virtual environment where the package was installed.
    1.  **Check Availability**: The user will type `mlip-auto` in their terminal and press enter.
    2.  **Get Help**: They will run `mlip-auto --help`. They will be asked to verify that a well-formatted help message is printed to the screen, showing the available commands like `run`, `train`, and `dashboard`.
    3.  **Run a "Dry Run"**: The user will be asked to run a command that does something trivial but confirms the application logic is accessible. For example, `mlip-auto run --input non_existent_file.yaml`. The test passes if the application runs and prints an informative, user-friendly error message like "Error: Input file 'non_existent_file.yaml' not found." rather than crashing with a long, unhandled Python traceback.
*   **Why this UAT is amazing for the user**: This series of scenarios provides the complete "new user" experience, from initial curiosity to a successful first interaction with the tool. It focuses on the critical moments that determine whether a user will adopt a new piece of software: Is it well-documented? Is it easy to install? Does it "just work"? A successful run through this UAT provides high confidence that the project is not only scientifically powerful but also professionally packaged and ready for a public release.

## 2. Behavior Definitions

The following Gherkin-style definitions describe the expected behavior of the system for the UAT scenarios.

### Scenario: Can a New User Understand the Project from its Documentation?

*   **GIVEN** a user has been provided with the URL to the MLIP-AutoPipe documentation website.

*   **WHEN** they read the home page.
*   **THEN** they should be able to accurately summarize the project's main purpose.

*   **WHEN** they navigate to the "Installation" page.
*   **THEN** the instructions for installing via `pip` should be clear and unambiguous.

*   **WHEN** they read the "Elastic Constants Tutorial".
*   **THEN** they should understand the basic workflow: create a minimal YAML file, run a CLI command, and get a final result.

### Scenario: Can a New User Easily Install the Software?

*   **GIVEN** a user has a clean Python virtual environment.
*   **AND** they have been provided with the path to the built Python wheel file.

*   **WHEN** they run the command `pip install <path_to_wheel_file>`.
*   **THEN** the installation process must complete with an exit code of 0.
*   **AND** there must be no error messages displayed during the installation.
*   **AND** the command `mlip-auto` should now be available in their shell's PATH.

### Scenario: Does the Packaged Application Run Correctly?

*   **GIVEN** the user has successfully installed the MLIP-AutoPipe package.

*   **WHEN** they run the command `mlip-auto --help`.
*   **THEN** a help message must be printed to the console.
*   **AND** this message must list the `run`, `train`, and `dashboard` sub-commands.

*   **WHEN** they run a command with an invalid input file, like `mlip-auto run --input does_not_exist.yaml`.
*   **THEN** the program must not crash with a Python traceback.
*   **AND** it must print a clear, user-friendly error message to the console, such as "Error: File not found".
*   **AND** it must exit with a non-zero exit code.
*   **AND** a final confirmation message should be displayed to the user, indicating that the UAT for the final cycle is complete and successful.
