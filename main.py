import sys
from pathlib import Path

# Add src to sys.path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from mlip_autopipec.main import app  # noqa: E402

if __name__ == "__main__":
    app()
