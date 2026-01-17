#!/usr/bin/env python
import shutil
from pathlib import Path

# This script mocks the behavior of pw.x for testing purposes.
# It copies a pre-generated, valid output file to the current working
# directory, simulating a successful Quantum Espresso run.

mock_output_file = Path(__file__).parent / "mock_espresso.pwo"
destination_file = Path.cwd() / "espresso.pwo"

shutil.copy(mock_output_file, destination_file)
