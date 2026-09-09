# Building a local desktop app

The frontend is optional and is designed for local lab machines.

## Run from source

```bash
pip install -e '.[all,frontend]'
neuro-import-gui
```

## PyInstaller sketch

```bash
pip install pyinstaller
pyinstaller --name NeuroSignalImporter --windowed -m neuro_importer_frontend.desktop_app
```

Large scientific dependencies can make packaged apps large. For lab use, running from a managed Python environment is usually easier.
