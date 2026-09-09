# Install and Run

Use this command for normal launches:

```bash
python3 run_neuro_signal_app.py
```

The launcher automatically:

1. Checks whether the local editable package is installed from this folder.
2. Checks required libraries for conversion, frontend, live replay, and file formats.
3. Runs `python -m pip install -e '.[all,frontend,live,dev]'` only if something is missing or the install points somewhere else.
4. Starts the local FastAPI server.
5. Opens `http://127.0.0.1:8787` when the server is ready.

It does **not** reinstall every time.

Force reinstall only when you really want to refresh the environment:

```bash
python3 run_neuro_signal_app.py --force-install
```

Other useful options:

```bash
python3 run_neuro_signal_app.py --port 8790
python3 run_neuro_signal_app.py --workspace ~/neuro_signal_app_workspace
python3 run_neuro_signal_app.py --no-browser
```
