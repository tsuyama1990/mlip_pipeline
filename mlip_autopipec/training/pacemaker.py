import re
import subprocess
import time
from pathlib import Path

from mlip_autopipec.config.schemas.training import TrainConfig, TrainingResult
from mlip_autopipec.training.config_gen import TrainConfigGenerator
from mlip_autopipec.training.dataset import DatasetBuilder


class PacemakerWrapper:
    """Wraps Pacemaker training process."""

    def __init__(self, executable: str = "pacemaker"):
        self.executable = executable

    def train(
        self,
        config: TrainConfig,
        dataset_builder: DatasetBuilder,
        config_gen: TrainConfigGenerator,
        work_dir: Path,
        generation: int = 0
    ) -> TrainingResult:
        """
        Executes the training loop.

        Steps:
        1. Export dataset.
        2. Generate config (input.yaml).
        3. Run pacemaker.
        4. Parse logs and result.
        """
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export Data
        data_path = dataset_builder.export(config, work_dir)

        # We need elements list for the config
        # Load dataset briefly to get elements? Or rely on DatasetBuilder?
        # DatasetBuilder doesn't return elements.
        # But we can assume elements are consistent in the dataset.
        # Let's peek at the file or ask DB?
        # Getting elements from DB is safer.

        # Assuming we can get all unique species from the database
        # For now, let's scan the first few atoms in the pickle file.
        # Or better, DatasetBuilder could return metadata.
        # I'll update DatasetBuilder to return list of elements?
        # Or just read the pickle back.

        import gzip
        import pickle
        with gzip.open(data_path, "rb") as f:
            # We only need to peek one frame if it's a single system type,
            # but for general robustness we should union all symbols.
            # Reading all might be slow if large.
            # But we just wrote it, so it's in memory cache likely.
            # Let's read it all.
            atoms_list = pickle.load(f)

        elements = set()
        for at in atoms_list:
            elements.update(at.get_chemical_symbols())
        sorted_elements = sorted(list(elements))

        # 2. Generate Config
        potential_name = f"potential_gen{generation}.yace"
        output_path = work_dir / potential_name

        input_yaml = config_gen.generate(config, data_path, output_path, sorted_elements)

        # 3. Run Pacemaker
        start_time = time.time()

        # We need to run inside work_dir so pacemaker outputs files there
        cmd = [self.executable, str(input_yaml.name)]

        # We capture output to parse RMSE
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                check=True
            )
            stdout = result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Pacemaker training failed: {e.stderr}") from e
        except FileNotFoundError:
             raise RuntimeError(f"Pacemaker executable '{self.executable}' not found.")

        end_time = time.time()
        training_time = end_time - start_time

        # 4. Parse Logs
        # We look for "RMSE energy:" and "RMSE forces:" in stdout.
        # Example output format (hypothetical, based on ACE/pacemaker logs):
        # "Final RMSE energy: 0.005 eV/atom"
        # "Final RMSE forces: 0.123 eV/A"

        rmse_energy = 0.0
        rmse_forces = 0.0

        # Simple regex parsing
        # Adjust patterns based on actual Pacemaker output
        # Usually it prints a table or specific lines at the end.

        # Pattern: "RMSE (energy) : 0.005"
        # Pattern: "RMSE (forces) : 0.123"

        re_energy = re.search(r"RMSE\s*\(energy\)\s*:\s*([\d\.eE\-\+]+)", stdout)
        if re_energy:
            rmse_energy = float(re_energy.group(1))

        re_forces = re.search(r"RMSE\s*\(forces\)\s*:\s*([\d\.eE\-\+]+)", stdout)
        if re_forces:
            rmse_forces = float(re_forces.group(1))

        # Verify output file exists
        if not output_path.exists():
            # Sometimes it adds .yace automatically or uses a different name
            # If not found, check if there is ANY .yace file and warn/use it
            yace_files = list(work_dir.glob("*.yace"))
            if yace_files:
                output_path = yace_files[0]
            else:
                raise FileNotFoundError(f"Output potential file {output_path} not found.")

        return TrainingResult(
            potential_path=output_path,
            rmse_energy=rmse_energy,
            rmse_forces=rmse_forces,
            training_time=training_time,
            generation=generation
        )
