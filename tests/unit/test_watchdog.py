import pytest

from mlip_autopipec.domain_models.config import DynamicsConfig
from mlip_autopipec.domain_models.enums import DynamicsType
from mlip_autopipec.dynamics.watchdog import UncertaintyWatchdog


def test_watchdog_enabled():
    config = DynamicsConfig(
        type=DynamicsType.LAMMPS,
        halt_on_uncertainty=True,
        max_gamma_threshold=5.0,
    )
    watchdog = UncertaintyWatchdog(config)

    commands = watchdog.get_commands("potential.yace", ["Fe", "Pt"])
    assert "compute pace all pace potential.yace Fe Pt" in commands
    assert "compute max_gamma all reduce max c_pace[1]" in commands
    # The variable should check if max_gamma > threshold
    assert "variable check_gamma equal c_max_gamma>5.0" in commands
    assert "fix halt_check all halt 1 v_check_gamma != 0 error 100" in commands


def test_watchdog_disabled():
    config = DynamicsConfig(
        type=DynamicsType.LAMMPS,
        halt_on_uncertainty=False,
    )
    watchdog = UncertaintyWatchdog(config)

    commands = watchdog.get_commands("potential.yace", ["Fe", "Pt"])
    assert commands == ""
