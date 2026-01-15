# CYCLE07: User Interface (CLI) (UAT.md)

## 1. Test Scenarios

User Acceptance Testing (UAT) for Cycle 7 is all about the command-line user experience. The goal is to ensure that interacting with the `mlip-auto` command is simple, intuitive, and robust. The user should feel that the CLI is a professional and reliable tool that guides them and provides clear feedback. These scenarios will be documented in a Jupyter Notebook, using `!` or `%%bash` cells to simulate and capture the output of the terminal commands, making the results easy to review and understand.

| Scenario ID   | Test Scenario                                             | Priority |
|---------------|-----------------------------------------------------------|----------|
| UAT-C7-001    | **"Happy Path" Workflow Execution**                       | **High**     |
| UAT-C7-002    | **Helpful Messages for Command-Line Errors**              | **High**     |
| UAT-C7-003    | **Graceful Handling of Invalid Configuration Files**      | **High**     |

---

### **Scenario UAT-C7-001: "Happy Path" Workflow Execution**

**(Min 300 words)**

**Description:**
This is the most fundamental UAT for the CLI. It tests the entire end-to-end user journey for a successful run. The user provides a valid configuration file to the `mlip-auto run` command and should see clear, positive feedback that the workflow has started correctly. This scenario builds the user's confidence in the tool's primary function and sets the baseline for a positive user experience.

**UAT Steps in Jupyter Notebook:**
1.  **Setup:** The notebook will first create a valid `input.yaml` file in the current directory. The contents of this file will be displayed in a code cell for clarity.
    ```yaml
    # In the notebook, creating the file:
    !echo '                               \
    project_name: "UAT Happy Path Test"    \
    target_system:                         \
      elements: ["Si"]                     \
      composition: { "Si": 1.0 }           \
      crystal_structure: "diamond"         \
    simulation_goal:                       \
      type: "elastic"                      \
    ' > happy_path.yaml
    ```
2.  **Execution:** A cell using the `%%bash` magic command will execute the CLI. To avoid running the entire (long) workflow, the test environment will be set up to use a mock `WorkflowManager` that simply prints a success message and exits.
    ```bash
    %%bash
    # Execute the command
    mlip-auto run happy_path.yaml
    ```
3.  **Verify Output:** The captured output of the cell will be displayed directly in the notebook. The user will see something like this, demonstrating the clear and formatted feedback from the `rich` library:
    ```
    ✅ Configuration file 'happy_path.yaml' loaded and validated.
    🚀 Starting MLIP-AutoPipe workflow for project: UAT Happy Path Test
    ...
    [Mock Workflow Manager] Workflow initiated successfully.
    ...
    🎉 Workflow completed successfully.
    ```
4.  **Explanation:** A markdown cell will summarise the result: "The `mlip-auto run` command successfully parsed the configuration file, initialized the workflow, and ran to completion. The clear, formatted output confirms each stage of the process, providing a positive and informative user experience."

---

### **Scenario UAT-C7-002: Helpful Messages for Command-Line Errors**

**(Min 300 words)**

**Description:**
A well-designed CLI should help the user when they make a mistake. This scenario focuses on the errors that can happen at the command-line level, before the application logic even starts. It will test Typer's built-in validation, ensuring the user gets immediate, helpful, and automatic feedback for common mistakes like forgetting an argument or pointing to a file that doesn't exist. This demonstrates a level of polish that makes the tool feel professional and easy to use.

**UAT Steps in Jupyter Notebook:**
1.  **Error Case 1: Missing Argument:** The notebook will execute the command without providing the required configuration file path.
    ```bash
    %%bash --out out --err err
    mlip-auto run
    ```
    The captured `stderr` will be displayed, showing Typer's automatic error message: "Error: Missing argument 'CONFIG_FILE'."
2.  **Error Case 2: File Not Found:** The notebook will execute the command with a path to a file that does not exist.
    ```bash
    %%bash --out out --err err
    mlip-auto run no_such_file.yaml
    ```
    The captured `stderr` will be displayed, showing another of Typer's helpful, built-in messages: "Error: Invalid value for 'CONFIG_FILE': File 'no_such_file.yaml' not found."
3.  **The "Help" Command:** The notebook will also demonstrate the auto-generated help menu.
    ```bash
    %%bash
    mlip-auto run --help
    ```
    The output will display the clean, readable help text, showing the usage, arguments, and options.
4.  **Explanation:** A markdown cell will explain the importance of these features: "The CLI is designed to be self-documenting and to fail fast. If you make a mistake in the command itself, the tool provides immediate and specific feedback, telling you exactly what you did wrong. The `--help` menu provides all the information you need to use the command correctly, ensuring a smooth user experience."

---

## 2. Behavior Definitions

### **UAT-C7-001: "Happy Path" Workflow Execution**

```gherkin
Feature: CLI Workflow Execution
  As a user,
  I want to start a workflow by calling the application from my terminal with a configuration file,
  So that I can easily launch my materials science projects.

  Scenario: User starts a workflow with a valid configuration file
    Given a valid `input.yaml` file exists on the filesystem
    When I execute the command `mlip-auto run <path_to_file>` in my terminal
    Then the application should start without command-line errors
    And the application's console output should confirm that the configuration was loaded successfully
    And the application should proceed to launch the backend workflow manager
    And the command should exit with a status code of 0 upon successful completion.
```

### **UAT-C7-002: Helpful Messages for Command-Line Errors**

```gherkin
Feature: CLI Usability and Help
  As a new user,
  I want the CLI to provide helpful error messages if I use it incorrectly,
  So that I can learn how to use the tool effectively.

  Scenario: User forgets to provide a config file path
    When I execute the command `mlip-auto run` without any arguments
    Then the application should exit immediately with a non-zero status code
    And it should print an error message to the console indicating that the 'CONFIG_FILE' argument is missing.

  Scenario: User provides a path to a non-existent file
    When I execute `mlip-auto run non_existent_file.yaml`
    Then the application should exit immediately with a non-zero status code
    And it should print an error message indicating that the file was not found.
```

### **UAT-C7-003: Graceful Handling of Invalid Configuration Files**

```gherkin
Feature: Robust Configuration File Parsing
  As a user,
  I want the application to check my configuration file for errors before starting a long run,
  So that I can avoid wasting time with a faulty configuration.

  Scenario: User provides a configuration file with a validation error
    Given an `input.yaml` file that contains a known validation error (e.g., composition sum is not 1.0)
    When I execute the `mlip-auto run` command with the path to this file
    Then the application should exit with a non-zero status code
    And the application should print a clear error message explaining the specific validation rule that failed
    And the application should **not** attempt to start the backend workflow.
```
