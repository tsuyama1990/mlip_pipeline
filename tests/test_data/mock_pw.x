#!/usr/bin/env python
import sys
from pathlib import Path

# This script mocks the behavior of pw.x for testing purposes.
# It reads the content of a pre-defined valid output file and prints it to stdout.

mock_output_content = (Path(__file__).parent / "mock_espresso.pwo").read_text()
sys.stdout.write(mock_output_content)
sys.exit(0)
