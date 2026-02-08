#!/usr/bin/env python
import sys
from pathlib import Path

# Add src to path so we can import the package without installation
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from mlip_autopipec.main import app  # noqa: E402

if __name__ == "__main__":
    app()
