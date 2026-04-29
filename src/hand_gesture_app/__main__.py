"""Entrypoint for `hand-gesture-gui` / `python -m hand_gesture_app`."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    gui = Path(__file__).resolve().parent.parent / "gui_version.py"
    if not gui.is_file():
        print(f"Error: missing {gui}", file=sys.stderr)
        sys.exit(1)
    runpy.run_path(str(gui), run_name="__main__")


if __name__ == "__main__":
    main()
