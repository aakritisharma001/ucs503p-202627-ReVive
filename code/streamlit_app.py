"""Streamlit Community Cloud entry point for the public synthetic demo."""

import os
from pathlib import Path
import runpy

os.environ["REVIVE_PUBLIC_DEMO"] = "0"
runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
