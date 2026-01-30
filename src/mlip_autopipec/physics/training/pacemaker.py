import logging
import re
import subprocess
from pathlib import Path

from jinja2 import Template

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.training import TrainingConfig

logger = logging.getLogger("mlip_autopipec")

INPUT_TEMPLATE = """
cutoff: {{ cutoff }}
seed: {{ seed }}
data:
  filename: {{ dataset_path }}
potential:
  deltaSplineBins: 0.001
  elements: {{ elements }}
  embeddings:
    ALL:
      npot: "FinnisSinclairShiftedScaled"
      fs_parameters: [1, 1, 1, 0.5]
      ndensity: 2
  bonds:
    ALL:
      radbase: ChebExpCos
      radparameters: [5.25]
      radcutoff: {{ cutoff }}
      seg_sub: [1, 1]
      bases:
        - {max_deg: {{ ladder_step[0] }}, type: Bond}
fit:
  loss: {kappa: {{ kappa }}, L1_coeffs: 1e-8, L2_coeffs: 1e-8}
  weighting: {type: EnergyBasedWeighting, DE: 1.0, DF: 1.0}
  optimizer: {type: BFGS, max_options: {maxiter: {{ max_epochs }} }}
  fit_cycles: 1
backend:
  evaluator: tensorpot
  batch_size: {{ batch_size }}
  display_step: 50
"""


class PacemakerRunner:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        config: TrainingConfig,
        dataset_path: Path,
        elements: list[str],
        cutoff: float = 5.0,
        seed: int = 42,
    ) -> Potential:
        """
        Run pace_train to generate a potential.
        """
        # Generate input.yaml
        template = Template(INPUT_TEMPLATE)
        input_content = template.render(
            cutoff=cutoff,
            seed=seed,
            dataset_path=str(dataset_path),
            elements=elements,
            kappa=config.kappa,
            max_epochs=config.max_epochs,
            batch_size=config.batch_size,
            ladder_step=config.ladder_step,
        )

        input_yaml_path = self.work_dir / "input.yaml"
        input_yaml_path.write_text(input_content)

        # Run pace_train
        cmd = ["pace_train", str(input_yaml_path)]
        logger.info(f"Running training: {' '.join(cmd)}")

        try:
            subprocess.run(
                cmd, check=True, cwd=self.work_dir, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"pace_train failed: {e.stderr}")
            raise RuntimeError(f"pace_train failed: {e.stderr}") from e
        except FileNotFoundError as e:
            raise RuntimeError("pace_train executable not found.") from e

        # Look for .yace files
        yace_files = list(self.work_dir.glob("*.yace"))
        if not yace_files:
            # Fallback for testing if exact name is not predictable or matched
            yace_path = self.work_dir / "potential.yace"
        else:
            yace_path = yace_files[0]

        if not yace_path.exists():
            raise RuntimeError("Training finished but no .yace file found.")

        # Parse log for RMSE
        log_path = self.work_dir / "log.txt"
        metrics: dict[str, float] = {}
        if log_path.exists():
            content = log_path.read_text()
            e_rmse = re.search(r"Energy RMSE:\s*([\d\.]+)", content)
            f_rmse = re.search(r"Force RMSE:\s*([\d\.]+)", content)
            if e_rmse:
                metrics["rmse_energy"] = float(e_rmse.group(1))
            if f_rmse:
                metrics["rmse_force"] = float(f_rmse.group(1))

        return Potential(
            path=yace_path, format="ace", elements=elements, metadata=metrics
        )

    def select_active_set(self, config: TrainingConfig, dataset_path: Path) -> Path:
        """
        Run pace_activeset to reduce the dataset size.
        """
        # Usage: pace_activeset <dataset> <output> --opts
        # We append _active.pckl.gzip to the stem of the original file,
        # but handle the double extension correctly
        name = dataset_path.name
        if name.endswith(".pckl.gzip"):
            base_name = name[: -len(".pckl.gzip")]
        else:
            base_name = dataset_path.stem

        output_path = self.work_dir / f"{base_name}_active.pckl.gzip"

        cmd = ["pace_activeset", str(dataset_path), str(output_path)]
        logger.info(f"Running active set selection: {' '.join(cmd)}")

        try:
            subprocess.run(
                cmd, check=True, cwd=self.work_dir, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"pace_activeset failed: {e.stderr}")
            raise RuntimeError(f"pace_activeset failed: {e.stderr}") from e
        except FileNotFoundError as e:
            raise RuntimeError("pace_activeset executable not found.") from e

        return output_path
