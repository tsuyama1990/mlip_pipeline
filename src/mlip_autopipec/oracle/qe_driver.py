from ase import Atoms
from ase.calculators.espresso import Espresso

try:
    from ase.calculators.espresso import EspressoProfile
except ImportError:
    EspressoProfile = None  # type: ignore[assignment, misc]

from mlip_autopipec.domain_models.config import OracleConfig


class QEDriver:
    """
    Driver for Quantum Espresso calculations using ASE.
    """

    def __init__(self, config: OracleConfig) -> None:
        self.config = config

    def get_calculator(self, atoms: Atoms) -> Espresso:
        """
        Configures and returns an ASE Espresso calculator based on the config.
        """
        # Define input parameters
        # Note: ASE Espresso uses a flat dict for some params if passed as kwargs,
        # or 'input_data' dict for namelists.
        # It's safer to use 'input_data' for namelist specific parameters.

        input_data = {
            "control": {
                "calculation": "scf",
                "tprnfor": True,
                "tstress": True,
                "disk_io": "low",  # Minimize I/O
            },
            "system": {
                "ecutwfc": self.config.encut,
                "occupations": "smearing",
                "smearing": "mv",
                "degauss": self.config.smearing_width,
            },
            "electrons": {
                "mixing_beta": self.config.mixing_beta,
                "conv_thr": 1e-6,  # Default convergence threshold
            },
        }

        # Pseudopotentials
        pseudos = self.config.pseudos

        # Prepare arguments
        kwargs = {
            "input_data": input_data,
            "pseudopotentials": pseudos,
            "kspacing": self.config.kspacing,
        }

        # Prepare pseudo_dir string or None
        pseudo_dir_str = str(self.config.pseudo_dir) if self.config.pseudo_dir else None

        if self.config.command:
            # Check for new ASE EspressoProfile (ASE > 3.22)
            if EspressoProfile is not None:
                # Use profile instead of command
                # EspressoProfile requires command and pseudo_dir
                kwargs["profile"] = EspressoProfile(  # type: ignore[no-untyped-call]
                    command=self.config.command, pseudo_dir=pseudo_dir_str
                )
            else:
                kwargs["command"] = self.config.command
                if pseudo_dir_str:
                    kwargs["pseudo_dir"] = pseudo_dir_str
        elif pseudo_dir_str:
            # If command not set, try setting pseudo_dir in kwargs
            kwargs["pseudo_dir"] = pseudo_dir_str

        # Initialize calculator
        return Espresso(**kwargs)  # type: ignore[no-untyped-call]
