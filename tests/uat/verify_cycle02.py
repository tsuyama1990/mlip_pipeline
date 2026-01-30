import sys
from pathlib import Path
import subprocess
import yaml

def main():
    print("Starting UAT Cycle 02...")

    # 1. Setup Dummy LAMMPS
    dummy_lammps = Path("fake_lammps.sh")
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
8
ITEM: BOX BOUNDS pp pp pp
0.0 5.43
0.0 5.43
0.0 5.43
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 1 1.35 1.35 1.35
3 1 2.71 2.71 0.0
4 1 4.07 4.07 1.35
5 1 2.71 0.0 2.71
6 1 4.07 1.35 4.07
7 1 0.0 2.71 2.71
8 1 1.35 4.07 4.07
"""
    script_content = f"""#!/bin/bash
echo "Simulating LAMMPS..."
# Args: -in in.lammps
# We just write dump.lammpstrj
echo "{dump_content}" > dump.lammpstrj
"""
    dummy_lammps.write_text(script_content)
    dummy_lammps.chmod(0o755)
    fake_cmd_abs = str(dummy_lammps.resolve())

    # 2. Initialize Config
    # We can use CLI 'init' but we need to modify it.
    subprocess.run(["uv", "run", "mlip-auto", "init"], check=True)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["lammps"]["command"] = fake_cmd_abs
    config["lammps"]["timeout"] = 5.0

    with open("config.yaml", "w") as f:
        yaml.dump(config, f)

    # 3. Run Cycle 02
    print("Running Cycle 02...")
    try:
        res = subprocess.run(
            ["uv", "run", "mlip-auto", "run-cycle-02"],
            check=True,
            capture_output=True,
            text=True
        )
        print(res.stdout)
        if "Simulation Completed: Status COMPLETED" in res.stdout:
            print("SUCCESS: Simulation completed.")
        else:
            print("FAILURE: output does not contain success message.")
            print(res.stderr)
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        print("FAILURE: Command exited with error.")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

    # 4. Verify Artifacts
    if Path("_work_md/job_one_shot/dump.lammpstrj").exists():
         print("SUCCESS: Artifacts found.")
    else:
         print("FAILURE: Artifacts missing.")
         sys.exit(1)

    print("UAT Cycle 02 Passed!")

    # Cleanup
    dummy_lammps.unlink()
    if Path("config.yaml").exists():
        Path("config.yaml").unlink()

if __name__ == "__main__":
    main()
