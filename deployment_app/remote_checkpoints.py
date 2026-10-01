"""Checkpoints that are too large to push to GitHub (> 100 MB) and are stored
as files of a GitHub Release instead. The app downloads each file listed here
automatically (once per server start, skipped when already present) before it
scans for checkpoints -- no manual upload needed. Full guide: README,
"Deploy on Streamlit Community Cloud".

How to fill it in:
  1. GitHub repo page -> "Releases" (right sidebar) -> "Create a new release".
     Tag e.g. "checkpoints-v1", any title. Drag the .pkl files into
     "Attach binaries" and click "Publish release".
  2. On the release page, right-click a file name -> "Copy link address".
     It looks like:
     https://github.com/<user>/<repo>/releases/download/<tag>/<file>.pkl
  3. Add one entry per file below. "filename" must follow the naming pattern
     checkpoint_reg_<TAG>.pkl / checkpoint_p2_<measured|selected>_<TAG>.pkl.

PRIVATE repository? Release files are then not downloadable without a token:
add GITHUB_TOKEN to the app's Secrets on Streamlit Cloud (see README). Never
write the token in this file.
"""
REMOTE_CHECKPOINTS = [
    # {"filename": "checkpoint_reg_No-Aug_Optuna.pkl",
    #  "url": "https://github.com/arifsani2015/Static-Equipment-Corrosion-and-Risk-Prediction-ML/releases/download/checkpoints-v1/checkpoint_reg_No-Aug_Optuna.pkl"},
    # {"filename": "checkpoint_p2_selected_No-Aug_Optuna.pkl",
    #  "url": "https://github.com/arifsani2015/Static-Equipment-Corrosion-and-Risk-Prediction-ML/releases/download/checkpoints-v1/checkpoint_p2_selected_No-Aug_Optuna.pkl"},
]
