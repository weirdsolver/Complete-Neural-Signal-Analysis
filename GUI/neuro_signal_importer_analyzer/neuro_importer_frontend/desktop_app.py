from __future__ import annotations

import sys


def main() -> None:
    try:
        from PySide6.QtWidgets import QApplication
        from neuro_importer_frontend.main_window import MainWindow
    except Exception as exc:  # pragma: no cover
        print("The desktop frontend requires PySide6. Install it with:")
        print("  pip install -e '.[frontend]'")
        print(f"Import error: {exc}")
        raise SystemExit(2)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
