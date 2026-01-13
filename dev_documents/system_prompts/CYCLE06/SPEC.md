# CYCLE06 Specification: The Final Polish

## 1. Summary

This document provides the detailed technical specification for CYCLE06, the final development cycle of the MLIP-AutoPipe project before its initial official release. The primary focus of this cycle is not on adding new features, but on ensuring the quality, robustness, usability, and deployability of the entire system. This "final polish" is critical for transforming the powerful prototype into a reliable, maintainable, and distributable piece of scientific software. The work in this cycle is divided into three main areas: **Comprehensive Documentation**, **Rigorous End-to-End Testing**, and **Simplified Deployment and Packaging**.

First, a complete and professional **documentation** suite will be created. This goes far beyond code comments. It will include:
*   **User Documentation**: Hosted online (e.g., using MkDocs or ReadTheDocs), explaining the concepts, installation, and usage of the tool with tutorials and examples for each of the advanced simulation goals.
*   **API Documentation**: Automatically generated from the Pydantic schemas and Python docstrings, providing a clear reference for developers who may wish to extend the system.
*   **Deployment Guides**: Detailed instructions for HPC administrators on how to deploy the system and its dependencies (like Redis and Quantum Espresso) in a cluster environment.

Second, a suite of **end-to-end (E2E) regression tests** will be developed. While previous cycles included integration tests for new features, this cycle will build a comprehensive test suite that runs the full workflow for each of the supported `simulation_goal` types. These tests will use small, well-understood physical systems and will be designed to run automatically as part of a Continuous Integration (CI) pipeline. Their purpose is to act as a safety net, ensuring that future code changes do not break the complex, interconnected workflows that have been built. Stress testing will also be performed to understand the system's limits and identify performance bottlenecks under heavy load.

Third, the project will be packaged for **easy deployment and distribution**. This involves creating containerization scripts (e.g., Docker for local testing, Apptainer/Singularity for HPC environments) that bundle the application and all its dependencies into a single, portable image. This dramatically simplifies installation for end-users, resolving the "dependency hell" that often plagues scientific software. The Python package itself will be properly configured in `pyproject.toml` for distribution via the Python Package Index (PyPI), allowing users to install it with a simple `pip install mlip-autopipec`.

By the end of CYCLE06, the MLIP-AutoPipe project will be a mature, well-documented, thoroughly tested, and easily deployable platform, ready for its first wave of users in the materials science community.

## 2. System Architecture

The architecture of the system itself remains largely unchanged in this cycle. The focus is on adding infrastructure for testing, documentation, and packaging around the existing application.

**File Structure for CYCLE06:**

New directories for documentation, E2E tests, and containerization will be added at the root level of the project.

```
mlip_autopipec/
├── src/
│   └── ... (existing application code)
├── tests/
│   ├── ... (existing unit and integration tests)
│   └── **e2e/**
│       ├── __init__.py
│       ├── **test_e2e_equilibrate.py**
│       ├── **test_e2e_diffusion.py**
│       └── **test_e2e_elastic.py**
├── **docs/**
│   ├── **index.md**          # Landing page for the documentation
│   ├── **installation.md**
│   ├── **usage/**
│   │   ├── **cli_reference.md**
│   │   └── **tutorial_diffusion.md**
│   ├── **developer/**
│   │   └── **api_reference.md**
│   └── **mkdocs.yml**        # Configuration for the documentation generator
├── **containers/**
│   ├── **Dockerfile**        # For local testing and development
│   └── **Apptainer.def**     # For HPC deployment
├── pyproject.toml            # Finalized for PyPI distribution
└── README.md                 # Polished and expanded
```

**Architectural Blueprint:**

1.  **Documentation Generation**:
    *   A static site generator, `MkDocs`, will be used to build the documentation.
    *   The `mkdocs.yml` file will define the site structure, theme, and navigation.
    *   Markdown files in the `docs/` directory will be written for user guides and tutorials.
    *   A tool like `mkdocstrings` will be configured to automatically pull docstrings from the Python source code and render them as a clean API reference in `api_reference.md`.
    *   The entire documentation site will be built into a static `site/` directory, which can be hosted on any web server or service like GitHub Pages.
2.  **Continuous Integration (CI) Pipeline**:
    *   A CI pipeline (e.g., using GitHub Actions) will be established.
    *   On every commit or pull request, the pipeline will automatically:
        a. Install all dependencies.
        b. Run the linters (`ruff`, `mypy`).
        c. Run all unit and integration tests.
        d. **Run the new E2E tests.** These may be run on a schedule (e.g., nightly) if they are too slow for every commit.
        e. (On merge to `main`) Build the documentation site to ensure it is up-to-date.
        f. (On creating a new release tag) Build the Python package and push it to PyPI.
3.  **Containerization**:
    *   The `Dockerfile` will define a multi-stage build. It will start from a base Python image, install all system and Python dependencies, and then copy the application code into the image. This creates a self-contained environment for running the tool.
    *   The `Apptainer.def` (formerly Singularity) file will serve a similar purpose but will be tailored for HPC environments, which often have different security constraints and module systems. It will provide a portable and reproducible environment for running MLIP-AutoPipe on a shared cluster.
4.  **Packaging**:
    *   The `pyproject.toml` file will be finalized. This includes adding metadata such as the author, license, project URL (pointing to the new documentation site), and defining the CLI entry points (e.g., `mlip-auto = mlip_autopipec.cli:app`). This configuration allows `uv` or `pip` to build a distributable wheel (`.whl`) and source distribution (`.tar.gz`) for PyPI.

This architecture provides the scaffolding necessary to support a mature, open-source scientific software project, ensuring its long-term quality and usability.

## 3. Design Architecture

The design for CYCLE06 focuses on the content of the documentation, the structure of the E2E tests, and the configuration of the packaging tools.

**Documentation Content Design (`docs/`):**

*   **`index.md`**: A high-level overview of what MLIP-AutoPipe is, its core philosophy, and who it is for. It will feature a compelling example.
*   **`installation.md`**: Clear, step-by-step instructions for three installation methods:
    1.  From PyPI using `pip` (the recommended method).
    2.  Using the Docker container.
    3.  For developers, cloning the source and installing in editable mode.
*   **`usage/cli_reference.md`**: An auto-generated or manually curated reference for all CLI commands and their options.
*   **`usage/tutorial_*.md`**: A series of narrative-driven tutorials, similar to the UATs, that walk a user through a complete workflow for each `simulation_goal`. They will include the minimal YAML input and show the expected output and how to interpret it.
*   **`developer/api_reference.md`**: The `mkdocstrings` output, providing a reference for the Pydantic schemas and the public methods of the main classes, intended for those who want to extend the software.

**End-to-End Test Design (`tests/e2e/`):**

*   **Philosophy**: Each E2E test will be self-contained and will test one full `simulation_goal` workflow. They will use the smallest, cheapest physical system possible to keep runtime down. They will use real, but mock-scale, external tools (e.g., a mock DFT calculator that returns pre-computed results from a lookup table).
*   **`test_e2e_equilibrate.py`**: Will run the full workflow for the `'equilibrate'` goal on a simple system like Aluminum. It will assert that the final lattice constant is correct. This formalizes the CYCLE04 UAT into a regression test.
*   **`test_e2e_diffusion.py`**: Will run the `'diffusion'` workflow. It will need to be carefully designed to be fast. It might use a 2D LJ system where diffusion is very fast. It will assert that the final calculated diffusion coefficient is within a reasonable tolerance of the expected value for the test system.
*   **`test_e2e_elastic.py`**: Will formalize the CYCLE05 UAT. It will run the `'elastic'` workflow for Silicon and assert that the final calculated elastic constants are correct.
*   These tests are the ultimate guardians of the system's stability. A failure in an E2E test indicates a major regression that must be fixed before a new release.

**Containerization and Packaging Design:**

*   **`Dockerfile`**: Will be optimized for small image size. It will use a multi-stage build to first compile any dependencies, then copy only the necessary runtime artifacts into the final image, excluding build tools and temporary files. It will define a default `ENTRYPOINT` to be `mlip-auto`, so a user can run `docker run mlip-autopipec run --help`.
*   **`pyproject.toml`**: The `[project.scripts]` section will be the primary focus. It will be configured to create the `mlip-auto` executable when the package is installed. The `[project.urls]` will be populated to point to the documentation and source repository. The list of `dependencies` will be carefully curated to be complete.

## 4. Implementation Approach

The implementation will involve three parallel streams of work that can be largely carried out independently.

**Step 1: Write the Documentation**
*   Set up MkDocs by creating the `docs/` directory and the initial `mkdocs.yml`.
*   Write the content for the user guide and tutorials as Markdown files. This is a significant writing task.
*   Go through the existing codebase and add high-quality docstrings to all public classes and methods, ensuring they are in a format that `mkdocstrings` can parse (e.g., Google or NumPy style).
*   Configure `mkdocs.yml` to use `mkdocstrings` to build the API reference.
*   Set up a simple web server locally (`mkdocs serve`) to preview the documentation as it is being written.

**Step 2: Build the End-to-End Test Suite**
*   Create the `tests/e2e/` directory.
*   For each `simulation_goal`, create a new test file.
*   The tests will use `pytest`. They will likely be complex, requiring significant setup (e.g., creating temporary run directories, writing config files) and teardown.
*   A key component will be a "mock DFT calculator" that can be used in place of Quantum Espresso to make the tests fast and deterministic. This mock calculator will read a small set of pre-computed results from a file and return them when called with the corresponding structure.
*   Integrate these tests into the CI pipeline (e.g., the GitHub Actions workflow file). Mark them as "slow" if necessary so they can be run on a different schedule from the faster unit tests.

**Step 3: Implement Packaging and Containerization**
*   Finalize the `pyproject.toml` file, ensuring all metadata is present and correct.
*   Build the package locally (`uv build`) and test installing the resulting wheel in a clean virtual environment to ensure the installation process works and the `mlip-auto` command is created correctly.
*   Write the `Dockerfile`. Build the image locally (`docker build .`) and test running it. Verify that the application can be executed from within the container.
*   Write the `Apptainer.def` file and test building the `.sif` image.
*   Set up the necessary secrets in the CI environment (e.g., a PyPI API token) and configure the CI pipeline to automatically publish to PyPI when a new release is tagged in git.

## 5. Test Strategy

The test strategy for CYCLE06 *is* the plan itself. The entire cycle is focused on creating the final layer of testing and validation for the project. The success of this cycle is measured by the quality and coverage of the documentation and the E2E test suite.

**Unit Testing Approach (Min 300 words):**

While the main focus is on E2E tests, some new utilities created for this cycle will require unit tests.

*   **Mock DFT Calculator**: The mock DFT calculator, which will be central to the E2E tests, must be unit-tested itself. We will create a test that loads a sample file of pre-computed results. We will then pass it an `ase.Atoms` object that exactly matches one of the inputs in the file and assert that it returns the correct, corresponding energy and forces. We will also test its behavior when it receives a structure that is *not* in its database, ensuring it raises an appropriate error. This guarantees that our E2E test infrastructure is reliable.
*   **Docstring Parsing**: While we are not testing the `mkdocstrings` tool itself, we can have a CI step that builds the documentation and checks for warnings. This acts as a unit test for our docstrings, ensuring they are correctly formatted and that all documented code elements can be found. Any warnings would indicate a broken link in the API documentation, which should be fixed.

**Integration Testing Approach (Min 300 words):**

The integration tests for this cycle are the End-to-End tests. They are the ultimate integration test, verifying the entire software stack from the user's command-line input to the final scientific result.

*   **CI Pipeline as a Test**: The CI pipeline itself is a form of integration test. A successful run of the entire pipeline (linting, unit tests, E2E tests, documentation build) is an integration test that verifies that all the different components of the development and deployment process are working together correctly.
*   **The E2E tests (`tests/e2e/`) are the primary deliverable.** Their design is the test strategy. As described in the Design section, each E2E test will:
    1.  **Define a User Story**: Start with a minimal YAML file, mimicking a real user.
    2.  **Execute the Full Stack**: Call the `mlip-auto` CLI, which triggers the Heuristic Engine, which in turn starts the `WorkflowManager`, which then dispatches jobs to the (mocked) backend.
    3.  **Assert the Final Scientific Outcome**: The test will not just check for completion. It will parse the final results and assert their physical correctness against known values. For example, `test_e2e_elastic.py` will assert `abs(c11_calculated - c11_known) < tolerance`.
*   **Testing the Deployment**: The final test is of the deployment process itself. The CI pipeline will build the Docker image. A subsequent step in the pipeline will then *use* that image to run a simple command, like `docker run <image_name> --help`. A successful execution of this command is an integration test that proves the containerization process is working correctly. Similarly, the CI pipeline will build the PyPI package and then attempt to install it in a fresh environment, testing the packaging and installation process. This ensures that the software we deliver to users is not just correct, but also correctly packaged.
