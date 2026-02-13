import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src is in python path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from ase import Atoms

from mlip_autopipec.core.factory import ComponentFactory
from mlip_autopipec.domain_models.config import GlobalConfig, TrainerConfig
from mlip_autopipec.domain_models.datastructures import Structure
from mlip_autopipec.domain_models.enums import TrainerType


def verify_cycle04() -> None:  # noqa: C901
    print("Starting Cycle 04 Verification...")  # noqa: T201

    # 1. Setup Config
    trainer_config = TrainerConfig(
        type=TrainerType.PACEMAKER,
        cutoff=4.5,
        order=2,
        basis_size=200,
        delta_learning="zbl",
        max_epochs=5
    )

    global_config = MagicMock(spec=GlobalConfig)
    global_config.trainer = trainer_config

    work_dir = Path("verification_output")
    work_dir.mkdir(exist_ok=True)

    factory = ComponentFactory(config=global_config)

    # Mock subprocess because we don't have pacemaker installed
    with patch("subprocess.run") as mock_run:
        def side_effect(cmd: list[str] | str, **kwargs: object) -> MagicMock:
            # Mock file creation based on command
            if isinstance(cmd, list):
                if "pace_collect" in cmd[0]:
                     try:
                         idx = cmd.index("--output")
                         out = Path(cmd[idx + 1])
                         out.touch()
                     except ValueError:
                         pass
                if "pace_train" in cmd[0]:
                     # Mock output potential
                     (work_dir / "output_potential.yace").touch()
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        # 2. Create Trainer
        try:
            trainer = factory.create_trainer(work_dir)
            print("✓ Trainer created successfully")  # noqa: T201
        except ValueError as e:
            print(f"✗ Failed to create trainer: {e}")  # noqa: T201
            return

        # 3. Prepare Data
        atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]])
        structures = [
            Structure(
                atoms=atoms,
                provenance="uat",
                energy=-5.0,
                forces=[[0.1, 0, 0], [-0.1, 0, 0]],
                stress=[0]*6
            )
        ]

        # 4. Train
        try:
            potential = trainer.train(structures)
            print("✓ Training executed successfully")  # noqa: T201
        except Exception as e:
            print(f"✗ Training failed: {e}")  # noqa: T201
            import traceback
            traceback.print_exc()
            return

        # 5. Verify Output
        if potential.format == "yace" and potential.path.exists():
            print("✓ Potential file verified")  # noqa: T201
        else:
            print("✗ Potential file missing or invalid format")  # noqa: T201

        # 6. Verify Config Generation
        input_yaml = work_dir / "input.yaml"
        if input_yaml.exists():
            content = input_yaml.read_text()
            if "cutoff: 4.5" in content and "zbl" in content:
                 print("✓ Input YAML configuration verified")  # noqa: T201
            else:
                 print("✗ Input YAML content mismatch")  # noqa: T201
                 print(content)  # noqa: T201
        else:
            print("✗ Input YAML not found")  # noqa: T201

    print("Cycle 04 Verification Complete!")  # noqa: T201

if __name__ == "__main__":
    verify_cycle04()
