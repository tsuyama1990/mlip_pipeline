"""
This is a temporary helper to generate a canonical mock output file.
It will be deleted after the file is generated.
"""

import os

import pytest
from ase.build import bulk
from ase.calculators.espresso import EspressoProfile

from mlip_autopipec.modules.dft import DFTFactory, QEInputGenerator, QEProcessRunner


# This requires a real pw.x executable in the path
# If you don't have one, this test will fail, but that's okay.
# We just need to run it once where pw.x is available to get the file.
@pytest.mark.skipif("pw.x" not in os.environ.get("PATH", ""), reason="pw.x not in PATH")
def test_generate_real_output(tmp_path):
    """
    Runs a real, minimal QE calculation to produce a valid output file.
    """
    profile = EspressoProfile(command="pw.x", pseudo_dir=None)
    input_generator = QEInputGenerator(profile=profile, pseudopotentials_path=None)
    process_runner = QEProcessRunner(profile=profile)

    factory = DFTFactory(input_generator=input_generator, process_runner=process_runner)

    # Use a single H atom for a very fast calculation
    atoms = bulk("H", "sc", a=10)  # Large cell to make it isolated

    # We need a dummy pseudopotential file for ASE to be happy
    pseudo_path = tmp_path / "H.pbe-rrkjus.UPF"
    pseudo_path.touch()
    factory.input_generator.pseudopotentials_path = tmp_path

    # The factory will fail because the pseudo is garbage, but it will
    # still produce an output file with the necessary structure info
    # that we can copy and edit.
    try:
        factory.run(atoms)
    except Exception:
        # We expect this to fail, but the output file should be there
        pass

    # The real output will be in a temp directory, this is hard to get.
    # Instead, we will manually run the command to get the output.
    from ase.io import write

    write(
        tmp_path / "espresso.pwi",
        atoms,
        format="espresso-in",
        pseudopotentials={"H": "H.pbe-rrkjus.UPF"},
        kpts=(1, 1, 1),
        ecutwfc=30,
        ecutrho=240,
    )

    command = f"cd {tmp_path} && pw.x < espresso.pwi > espresso.pwo"
    os.system(command)

    print(
        f"File generated at {tmp_path / 'espresso.pwo'}. Copy its content to tests/test_data/mock_espresso.pwo"
    )

    # This will fail the test run, which is fine.
    assert False
