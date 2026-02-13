from ase import Atoms

from mlip_autopipec.domain_models.config import OracleConfig
from mlip_autopipec.domain_models.enums import OracleType
from mlip_autopipec.oracle.qe_driver import QEDriver


def test_qe_driver_calculator_config() -> None:
    config = OracleConfig(
        type=OracleType.DFT,
        encut=60.0,
        kspacing=0.04,
        mixing_beta=0.5,
        smearing_width=0.02,
        pseudos={"Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF"},
        command="mpirun -np 4 pw.x",
    )

    driver = QEDriver(config)
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[2, 2, 2], pbc=True)

    calc = driver.get_calculator(atoms)

    # Check ASE Espresso parameters
    params = calc.parameters

    # NOTE: ASE Espresso calculator stores input_data in 'input_data' key
    # or flattens it depending on initialization.
    # When initialized with 'input_data' dict, it might keep it there.
    # We should check if 'ecutwfc' is in params directly or in params['input_data']['system']

    # ASE behavior:
    # If using 'input_data', it's stored in calc.parameters['input_data'] usually.
    # Or flattened?

    if "input_data" in params:
        input_data = params["input_data"]
        assert input_data["system"]["ecutwfc"] == 60.0
        assert input_data["electrons"]["mixing_beta"] == 0.5
        assert input_data["system"]["degauss"] == 0.02
        assert input_data["control"]["tprnfor"] is True
        assert input_data["control"]["tstress"] is True
    else:
        # If flattened (some versions/usages)
        assert params["ecutwfc"] == 60.0
        assert params["mixing_beta"] == 0.5
        assert params["degauss"] == 0.02
        assert params["tprnfor"] is True
        assert params["tstress"] is True

    # K-points
    assert params["kspacing"] == 0.04

    # Pseudos
    assert params["pseudopotentials"]["Si"] == "Si.pbe-n-kjpaw_psl.1.0.0.UPF"

    # Command
    if hasattr(calc, "profile") and calc.profile:
        # New ASE EspressoProfile
        # It stores command as 'command' attribute (string)
        assert calc.profile.command == "mpirun -np 4 pw.x"
    else:
        # Legacy
        assert calc.command == "mpirun -np 4 pw.x"  # type: ignore[attr-defined]
