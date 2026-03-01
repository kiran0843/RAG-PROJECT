"""Launch the Streamlit RAG app."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
app_path = PROJECT_ROOT / "src" / "app" / "main.py"

if not app_path.exists():
    print("App not found:", app_path)
    sys.exit(1)

sys.exit(
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"],
        cwd=str(PROJECT_ROOT),
    ).returncode
)
