# Cycle 04 User Acceptance Test (UAT)

## 1. Test Scenarios

### Scenario 1: Dataset Serialization
**Priority**: P0 (Critical)
**Description**: Verify that `DFTResult` objects can be serialized to the specific `pandas.DataFrame` pickle format required by Pacemaker.
**Steps**:
1.  Create 10 `DFTResult` objects with fake data.
2.  Invoke `DatasetManager.save(results, "train.pckl.gzip")`.
3.  Verify the file exists.
4.  Load it back using `pandas.read_pickle`.
5.  Verify columns: `energy`, `forces`, `stress`, `ase_atoms`.

### Scenario 2: Mock Training Execution
**Priority**: P1 (High)
**Description**: Verify that the Mock Trainer simulates the training process and produces a dummy potential file.
**Steps**:
1.  Configure `config.yaml` with `trainer.type: mock`.
2.  Provide a dummy dataset file.
3.  Run `mlip-runner train config.yaml`.
4.  Inspect `potentials/generation_001.yace`.
5.  Verify the file is created and contains dummy content (e.g., "Mock Potential").

### Scenario 3: Active Set Selection (Mock Logic)
**Priority**: P2 (Medium)
**Description**: Verify that the system attempts to select an active set before training.
**Steps**:
1.  Configure `config.yaml` with `trainer.activeset_selection: true`.
2.  Run the training command.
3.  Check logs for "Selecting Active Set using MaxVol...".
4.  Verify that `activeset.pckl.gzip` is created (mocked).

## 2. Behavior Definitions (Gherkin)

### Feature: Potential Training

**Scenario**: Successful Dataset Creation
    **Given** a list of valid DFT results
    **When** the Dataset Manager saves them
    **Then** a `.pckl.gzip` file should serve as input for Pacemaker
    **And** the file should be readable by Pandas

**Scenario**: Mock Training Loop
    **Given** a configured Mock Trainer
    **When** the Orchestrator requests training
    **Then** a `potential.yace` file should be generated in the output directory
    **And** the training time should be negligible

**Scenario**: Configuration Generation
    **Given** a user configuration specifying `prior_type: zbl`
    **When** the Trainer generates `input.yaml` for Pacemaker
    **Then** the YAML should contain a `prior` section with ZBL parameters
    **And** the cutoff radius should match the global configuration
