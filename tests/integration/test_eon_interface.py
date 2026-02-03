from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.config.config_model import Config, EonConfig, StructureGenConfig, TrainingConfig
from mlip_autopipec.domain_models.exploration import AKMCTask
from mlip_autopipec.domain_models.structures import CandidateStructure
from mlip_autopipec.physics.structure_gen.explorer import AdaptiveExplorer


@pytest.fixture
def akmc_config(tmp_path: Path) -> Config:
    conf = MagicMock(spec=Config)
    conf.exploration = StructureGenConfig(
        strategy="adaptive",
        parameters={}
    )
    conf.training = MagicMock(spec=TrainingConfig)
    conf.training.dataset_path = tmp_path / "seed.xyz"
    conf.eon = EonConfig()

    # Create seed file
    atoms = Atoms("Cu", positions=[[0,0,0]], cell=[5,5,5])
    from ase.io import write
    write(conf.training.dataset_path, atoms)

    return conf

def test_adaptive_explorer_akmc_integration(akmc_config: Config, tmp_path: Path) -> None:
    # Setup Policy to return AKMC task
    with patch("mlip_autopipec.physics.structure_gen.policy.AdaptivePolicy.decide_strategy") as mock_policy:
        mock_policy.return_value = [AKMCTask()]

        explorer = AdaptiveExplorer(akmc_config)

        potential_path = tmp_path / "pot.yace"
        potential_path.touch()

        # Mock EonWrapper.run_akmc to avoid running real subprocess
        with (
            patch("mlip_autopipec.physics.dynamics.eon_wrapper.EonWrapper.run_akmc", return_value=0) as mock_run,
            patch.object(explorer, "_collect_eon_results") as mock_collect,
        ):
            mock_collect.return_value = [MagicMock(spec=CandidateStructure)]

            candidates = explorer.explore(potential_path, tmp_path)

            assert len(candidates) == 1
            mock_run.assert_called_once()
            mock_collect.assert_called_once()

            # Check arguments
            args, _ = mock_run.call_args
            assert args[0] == potential_path
            # args[1] is seed atoms, hard to compare directly if object copy, but we can check type
            assert isinstance(args[1], Atoms)
