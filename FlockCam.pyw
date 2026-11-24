"""Double-click this file to open FlockCam (no command prompt window).

It runs the app with the project's .venv if one exists, so the double-click
uses the same Python and packages you installed with `pip install -e .`.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # keep data/ and models/ inside the project folder

venv_python = ROOT / ".venv" / ("Scripts/pythonw.exe" if os.name == "nt" else "bin/python")
if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
    subprocess.Popen([str(venv_python), str(Path(__file__).resolve())], cwd=ROOT)
    sys.exit(0)

sys.path.insert(0, str(ROOT / "src"))
from flockcam.gui.app import main  # noqa: E402

main([])
