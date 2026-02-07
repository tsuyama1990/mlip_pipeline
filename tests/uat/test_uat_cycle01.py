from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.factory import (
    create_dynamics,
    create_generator,
    create_oracle,
    create_selector,
    create_trainer,
    create_validator,
)
from mlip_autopipec.main import app

runner = CliRunner()

def test_scenario_01_system_initialisation(tmp_path: Path) -> None:
    """
    SCENARIO 01: System Initialisation & Configuration
    Verify that the system can correctly parse a configuration file and initialise the necessary components.
    """
    workdir = tmp_path / "scenario_01"
    config_data = {
        "workdir": str(workdir),
        "max_cycles": 2,
        "oracle": {"type": "mock", "noise_level": 0.02},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"}
    }
    config_file = tmp_path / "config_mock.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    # Run CLI
    result = runner.invoke(app, ["run", str(config_file)])

    assert result.exit_code == 0
    # Check logs for initialization messages
    assert "Configuration loaded successfully" in result.stdout or "Configuration loaded successfully" in result.stderr
    assert "Initialised MockOracle" in result.stdout or "Initialised MockOracle" in result.stderr
    assert "Initialised MockTrainer" in result.stdout or "Initialised MockTrainer" in result.stderr

def test_scenario_02_mock_execution_loop(tmp_path: Path) -> None:
    """
    SCENARIO 02: Mock Execution Loop
    Verify that the Mock components interact correctly through the defined interfaces.
    """
    workdir = tmp_path / "scenario_02"
    if not workdir.exists():
        workdir.mkdir()

    # Manual instantiation logic as per UAT description
    # 1. Generator setup

    # Setup Config Object manually to test factories
    from mlip_autopipec.domain_models.config import (
        MockDynamicsConfig,
        MockGeneratorConfig,
        MockOracleConfig,
        MockSelectorConfig,
        MockTrainerConfig,
        MockValidatorConfig,
    )

    gen = create_generator(MockGeneratorConfig(initial_count=5))
    oracle = create_oracle(MockOracleConfig(noise_level=0.1))
    trainer = create_trainer(MockTrainerConfig())
    dynamics = create_dynamics(MockDynamicsConfig())
    validator = create_validator(MockValidatorConfig())
    selector = create_selector(MockSelectorConfig())

    # 2. Execution Flow
    # Generate structures
    structures = list(gen.generate(5, workdir))
    assert len(structures) == 5
    assert all(s.energy is None for s in structures)

    # Label data using Oracle
    labeled_structures = list(oracle.compute(structures))
    assert len(labeled_structures) == 5
    assert all(s.energy is not None for s in labeled_structures)

    # Train potential
    potential = trainer.train(labeled_structures, workdir)
    assert potential.path.exists()

    # Run Dynamics exploration
    exploration_result = dynamics.run(potential, labeled_structures[:1], workdir)
    assert len(exploration_result.structures) > 0

    # Select candidates
    candidates = exploration_result.structures
    selected = list(selector.select(candidates, 1))
    assert len(selected) <= 1

    # Validate potential
    val_result = validator.validate(potential, labeled_structures, workdir)
    assert isinstance(val_result.passed, bool)
