"""User Acceptance Test for Cycle 01: Core Functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from ase import Atoms
from ase.db import connect

from mlip_autopipec.config.system import SystemConfig
from mlip_autopipec.config.user import UserConfig
from mlip_autopipec.data.database import DatabaseManager
from mlip_autopipec.modules.dft_factory import DFTFactory

# A canonical, complete Quantum Espresso output that ASE can parse
SAMPLE_QE_OUTPUT = """
     Program PWSCF v.6.5 starts on 10Jan2024 at 10:10:10
     bravais-lattice index     = 0
     lattice parameter (a_0)   = 18.897261  a.u.
     celldm(1)=   18.897261
     number of atoms/cell      = 1
     number of atomic types    = 1
     crystal axes: (cart. coord. in units of a_0)
               a(1) = (   0.529177   0.000000   0.000000 )
               a(2) = (   0.000000   0.529177   0.000000 )
               a(3) = (   0.000000   0.000000   0.529177 )
     site n.     atom                  positions (alat units)
         1           Ni          tau(   1) = (   0.0000000   0.0000000   0.0000000  )
!    total energy              =      -1.00000000 Ry
     Forces acting on atoms (Ry/au):

     atom    1 type  1   force =     0.100000000   0.200000000   0.300000000

     Total force =     0.374166     Total SCF correction =     0.000000
     total   stress  (Ry/bohr**3)     (kbar)     P=   -0.00
      0.00000000   0.00000000   0.00000000
      0.00000000   0.00000000   0.00000000
      0.00000000   0.00000000   0.00000000
     JOB DONE.
"""


def main() -> None:
    """Execute the UAT scenario for Cycle 01."""
    print("--- Starting UAT for Cycle 01: Core Functionality ---")

    # Part 1: The Power of Schemas
    print("\n--- Part 1: Verifying Schema Validation ---")
    user_data = {
        "target_system": {"elements": ["Ni"], "composition": {"Ni": 1.0}},
        "simulation_goal": "elastic",
    }
    UserConfig(**user_data)
    print("✓ Successfully created UserConfig from valid data.")

    system_config = SystemConfig(
        dft={
            "pseudopotentials": {"Ni": "Ni.pbe-n-rrkjus_psl.1.0.0.UPF"},
            "system": {"nat": 1, "ntyp": 1, "ecutwfc": 60, "nspin": 2},
        }
    )
    print("✓ Successfully created a SystemConfig.")

    # Part 2: Executing a DFT Calculation
    print("\n--- Part 2: Executing a Mocked DFT Calculation ---")
    atoms = Atoms("Ni", positions=[(0, 0, 0)], cell=[10, 10, 10], pbc=True)
    factory = DFTFactory(system_config)

    with patch.object(factory.process_runner, "execute") as mock_execute:

        def side_effect(input_path: Path, output_path: Path) -> None:
            output_path.write_text(SAMPLE_QE_OUTPUT)

        mock_execute.side_effect = side_effect
        result_atoms = factory.run(atoms)

    print("✓ Mocked DFT run completed successfully.")
    assert "energy" in result_atoms.calc.results
    assert "forces" in result_atoms.calc.results
    print("✓ Parsed results contain 'energy' and 'forces'.")

    # Part 3: Data Persistence
    print("\n--- Part 3: Verifying Data Persistence ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "uat.db"
        db_manager = DatabaseManager(db_path)
        metadata = {"config_type": "uat_c01", "custom_id": "abc"}

        db_manager.write_calculation(result_atoms, metadata=metadata)
        print(f"✓ Wrote calculation to temporary database at {db_path}")

        # Verify by reading back
        conn = connect(db_path)  # type: ignore[no-untyped-call]
        row = conn.get(1)
        assert row.key_value_pairs["mlip_config_type"] == "uat_c01"
        assert row.key_value_pairs["mlip_custom_id"] == "abc"
        print("✓ Custom metadata was successfully persisted.")

    print("\n--- UAT for Cycle 01 Completed Successfully ---")


if __name__ == "__main__":
    main()
