from pathlib import Path

import yaml
from ase.db import connect

from .modules.a_generator import generate_structures
from .modules.b_explorer import SurrogateExplorer
from .modules.c_dft_factory import run_qe_calculation
from .modules.d_trainer import PacemakerTrainer
from .schemas.dft import DFTInput
from .schemas.system_config import DFTParams, GeneratorParams, SystemConfig
from .schemas.user_config import UserConfig
from .settings import settings
from .utils.logging import get_logger
from .utils.qe_utils import get_kpoints, get_sssp_recommendations

logger = get_logger(__name__)


def run_pipeline(config_path: str) -> None:
    """
    Runs the MLIP-AutoPipe pipeline.
    """
    logger.info(f"Loading configuration from: {config_path}")
    with Path(config_path).open() as f:
        user_config_dict = yaml.safe_load(f)
    user_config = UserConfig.model_validate(user_config_dict)

    # This is a placeholder for the heuristic engine
    generator_params = GeneratorParams(
        sqs_supercell_size=[],
        strain_magnitudes=[
            -0.05,
            -0.02,
            0,
            0.02,
            0.05,
        ],  # TODO: make configurable
        rattle_standard_deviation=0,
    )
    system_config = SystemConfig(
        user_config=user_config,
        dft_params=DFTParams(
            pseudopotentials={},
            cutoff_wfc=60,
            k_points=(8, 8, 8),
            smearing="gauss",
            degauss=0.01,
            nspin=1,
        ),
        generator_params=generator_params,
        surrogate_config=user_config.surrogate_config,
        trainer_config=user_config.trainer_config,
    )
    structures = generate_structures(system_config)

    explorer = SurrogateExplorer(system_config.surrogate_config)
    selected_structures = explorer.select_structures(structures)

    dft_params = DFTParams(
        pseudopotentials=get_sssp_recommendations(selected_structures[0]),
        cutoff_wfc=60,
        k_points=get_kpoints(selected_structures[0]),
        smearing="gauss",
        degauss=0.01,
        nspin=1,
    )
    system_config.dft_params = dft_params

    if settings.qe_command:
        db_path = f"{user_config.project_name}.db"
        logger.info(f"Writing results to {db_path}")
        with connect(db_path) as db:  # type: ignore[no-untyped-call]
            for atoms in selected_structures:
                try:
                    dft_input = DFTInput(atoms=atoms, dft_params=dft_params)
                    dft_output = run_qe_calculation(dft_input)
                    db.write(atoms, data=dft_output.model_dump())
                except Exception:
                    logger.exception("Failed to run DFT calculation for a structure")
                    raise
    else:
        logger.warning("`qe_command` not set, skipping DFT calculations.")


def train_pipeline(config_path: str, database_path: str, output_dir: Path) -> None:
    """
    Trains a new MLIP using a generated dataset.
    """
    logger.info(f"Loading configuration from: {config_path}")
    with Path(config_path).open() as f:
        user_config_dict = yaml.safe_load(f)
    user_config = UserConfig.model_validate(user_config_dict)

    trainer = PacemakerTrainer(user_config.trainer_config)
    trainer.train_potential(database_path, output_dir)
