"""Entry point.  Run with ``python -m simulator`` from the repository root."""

from __future__ import annotations

import os
import sys

# Allow the project's own analysis modules to be imported when the simulator is
# launched from somewhere other than the repository root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    from PySide6 import QtWidgets
    from simulator.ui.main_window import MainWindow, DARK_QSS

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Brillouin Sample Simulator")
    app.setStyleSheet(DARK_QSS)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
