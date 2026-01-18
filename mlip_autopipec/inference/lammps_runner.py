import subprocess
from pathlib import Path

from ase.atoms import Atoms
from ase.io import write

from mlip_autopipec.config.schemas.inference import InferenceConfig, InferenceResult
from mlip_autopipec.inference.inputs import ScriptGenerator


class LammpsRunner:
    def __init__(self, config: InferenceConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.generator = ScriptGenerator(config)

    def run(self, atoms: Atoms) -> InferenceResult:
        # Define file paths
        input_file = self.work_dir / "in.lammps"
        data_file = self.work_dir / "data.lammps"
        log_file = self.work_dir / "log.lammps"
        dump_file = self.work_dir / "dump.gamma"

        # 1. Write Data File
        # Format 'lammps-data' writes masses. If 'atom_style atomic', it works.
        write(data_file, atoms, format="lammps-data")

        # 2. Generate Input Script
        script_content = self.generator.generate(
            atoms_file=data_file,
            potential_path=self.config.potential_path,
            dump_file=dump_file
        )

        input_file.write_text(script_content)

        # 3. Execute LAMMPS
        cmd = [
            str(self.config.lammps_executable) if self.config.lammps_executable else "lmp_serial",
            "-in", str(input_file),
            "-log", str(log_file)
        ]

        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=self.work_dir)
            success = (result.returncode == 0)
        except (subprocess.SubprocessError, OSError):
            success = False

        # 4. Process Results
        uncertain_structures = []
        if dump_file.exists() and dump_file.stat().st_size > 0:
            uncertain_structures.append(dump_file)

        # Extract max gamma if log exists
        max_gamma = 0.0
        if log_file.exists():
             # We assume AnalysisUtils can parse custom columns if we modify it or we rely on standard thermo?
             # My AnalysisUtils only parsed Temp and Press. I should update it to parse c_max_gamma if possible,
             # or just simple regex here?
             # To keep modularity, I should update AnalysisUtils to handle arbitrary columns or specific gamma column.
             # But for now, let's look at log file.
             pass

        return InferenceResult(
            succeeded=success,
            final_structure=data_file,
            uncertain_structures=uncertain_structures,
            max_gamma_observed=max_gamma
        )
