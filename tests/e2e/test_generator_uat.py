
import yaml
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.core.database import DatabaseManager

runner = CliRunner()

def test_uat_2_1_generate_sqs_alloy(tmp_path):
    """
    Scenario 2.1: Generate SQS Alloy Structures
    """
    db_path = tmp_path / "mlip.db"

    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    config_data = {
        "target_system": {
            "name": "FeNi",
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.5, "Ni": 0.5},
            "crystal_structure": "fcc"
        },
        "generator_config": {
            "sqs": {"enabled": True, "supercell_size": [2, 2, 2]},
            "distortion": {"enabled": False},
            "defects": {"enabled": False},
            "number_of_structures": 10
        },
        "dft": {
            "pseudopotential_dir": str(pseudo_dir),
            "ecutwfc": 40.0,
            "kspacing": 0.15
        },
        "runtime": {
            "database_path": str(db_path),
            "work_dir": str(tmp_path / "work")
        }
    }

    config_file = tmp_path / "input.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    runner.invoke(app, ["db", "init", "--config", str(config_file)])

    result = runner.invoke(app, ["generate", "--config", str(config_file)])
    print(f"STDOUT: {result.stdout}")
    assert result.exit_code == 0, f"Generate failed: {result.stdout}"

    assert "Generated 1 structures" in result.stdout

    with DatabaseManager(db_path) as db:
        count = db.count()
        assert count == 1
        # Check config_type via selection, as it might not be in atoms.info
        assert db.count(selection="config_type=sqs") == 1

def test_uat_2_2_apply_strain(tmp_path):
    """
    Scenario 2.2: Apply Elastic Strain
    """
    db_path = tmp_path / "mlip.db"
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    config_data = {
        "target_system": {
            "name": "Al",
            "elements": ["Al"],
            "composition": {"Al": 1.0},
            "crystal_structure": "fcc"
        },
        "generator_config": {
            "sqs": {"enabled": False},
            "distortion": {
                "enabled": True,
                "strain_range": [-0.1, 0.1],
                "n_strain_steps": 5,
                "n_rattle_steps": 0
            },
            "number_of_structures": 100
        },
        "dft": {
            "pseudopotential_dir": str(pseudo_dir),
            "ecutwfc": 30.0,
            "kspacing": 0.15
        },
        "runtime": {"database_path": str(db_path), "work_dir": str(tmp_path)}
    }

    config_file = tmp_path / "input.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    runner.invoke(app, ["db", "init", "--config", str(config_file)])
    result = runner.invoke(app, ["generate", "--config", str(config_file)])
    print(f"STDOUT: {result.stdout}")
    assert result.exit_code == 0

    with DatabaseManager(db_path) as db:
        count = db.count()
        print(f"DB Count: {count}")
        # Base (1) + 4 strained = 5 structures
        assert count == 5

def test_uat_2_3_defect_vacancy(tmp_path):
    """
    Scenario 2.3: Defect Generation
    """
    # Use SQS enabled to ensure supercell
    db_path = tmp_path / "mlip.db"
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    config_data = {
        "target_system": {
            "name": "Cu",
            "elements": ["Cu"],
            "composition": {"Cu": 1.0},
            "crystal_structure": "fcc"
        },
        "generator_config": {
            "sqs": {"enabled": True, "supercell_size": [2, 2, 2]},
            "distortion": {"enabled": False},
            "defects": {
                "enabled": True,
                "vacancies": True,
                "interstitials": False
            },
            "number_of_structures": 10
        },
        "dft": {"pseudopotential_dir": str(pseudo_dir), "ecutwfc": 30, "kspacing": 0.15},
        "runtime": {"database_path": str(db_path), "work_dir": str(tmp_path)}
    }

    config_file = tmp_path / "input.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    runner.invoke(app, ["db", "init", "--config", str(config_file)])
    result = runner.invoke(app, ["generate", "--config", str(config_file)])
    print(f"STDOUT: {result.stdout}")
    assert result.exit_code == 0

    with DatabaseManager(db_path) as db:
        count = db.count()
        print(f"DB Count: {count}")
        # SQS (primitive * 8 = 8 atoms) -> Base
        # Vacancies -> 8 structures
        # Total 1 + 8 = 9 structures.
        assert count == 9

        # Check vacancies exist
        assert db.count(selection="config_type=vacancy") == 8
