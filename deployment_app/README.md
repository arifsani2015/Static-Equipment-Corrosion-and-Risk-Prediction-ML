# Corrosion Rate & Risk Prediction — Deployment App

Streamlit app for the models of **Experiment 1** (corrosion-rate regression) and **Experiment 2**
(API RP 581 risk classification). It runs on your own computer or as a permanent web link on
Streamlit Community Cloud.

## Pages
| Page | What it does |
|---|---|
| ⚙️ **Settings** | Checkpoints (automatic or manual upload), scenario & model selection with quality metrics, editable input schema, library-version check |
| 📈 **Corrosion Rate Prediction** | Predicts CR with an 80% interval, a confidence level and the 5 most similar real equipment |
| ⚠️ **Risk Prediction** | Predicts the risk class and shows the probability of **every** class |
| 🔗 **Combined (CR → Risk)** | For equipment without CR: predicts CR first, then risk (same flow as Experiment 3) |

Every prediction page supports **manual input** (one equipment) and **batch upload** (downloadable Excel
template). Every result — single, history or batch — can be downloaded as **.xlsx**.
(Page files live in `app_pages/`.)

## Checkpoints
The app needs the files written by the experiments:

| Type | Name pattern | Example |
|---|---|---|
| Corrosion rate (Experiment 1) | `checkpoint_reg_{TAG}.pkl` | `checkpoint_reg_No-Aug_Optuna.pkl` |
| Risk (Experiment 2) | `checkpoint_p2_{measured\|selected\|estimated}_{TAG}.pkl` | `checkpoint_p2_selected_No-Aug_Optuna.pkl` |

`{TAG}` can be anything; the rest of the pattern must match exactly. A checkpoint reaches the app in one of
three ways (all can be combined):
1. **`checkpoints/` folder** next to `app.py` — found automatically (also in sub-folders). Extra folders can be
   listed in the `CHECKPOINT_DIRS` environment variable.
2. **`remote_checkpoints.py`** — for files too large for git (see "Large checkpoints" below); downloaded
   automatically at start-up.
3. **Manual upload** on the Settings page — always replaces the automatically loaded checkpoint of the same type.

The input features of each model are read from its own `.pkl`, so checkpoints trained with a different set of
features work without any code change. Default models: the scenario with the lowest cross-validated RMSE
(preferring the CR type of the risk model) and its best model; any scenario/model can be chosen on the
Settings page. Quality labels (rule of thumb): corrosion rate R²(CV) ≥ 0.75 Good, ≥ 0.50 Fair, else Weak;
risk F1-macro(CV) ≥ 0.75 Good, ≥ 0.60 Fair, else Weak.

## Corrosion-rate confidence
Each CR prediction comes with: an **80% prediction interval** (from the model's test-set errors, conservative
split-conformal quantiles), the **model spread** (std/mean of all models), the **similarity to known equipment**
(distance to the 5 nearest real equipment: Inside ≤ 90th percentile, Borderline ≤ 99th, Outside), and the
**5 most similar real equipment** with their measured CR. Confidence level: Outside → Low; otherwise +1 Inside,
+1 spread ≤ 15% (−1 if > 35%), +1 prediction within the range of similar equipment; ≥ 2 High, ≥ 0 Medium, else Low.
Risk confidence is the probability of the predicted class (≥ 70% High, 50–70% Medium, < 50% Low).

## Input schema and batch template
Settings → section 4: display label, Excel header, default value, extra categories and show/hide are editable per
column (`cr` cannot be hidden). The batch template has the sheets **Input** (fill this), **Column guide**,
**Example** and **Instructions**; extra columns such as `No` or `Equipment/Pipe Name` are kept in the results.
Result tables show `#` (rank in the table) and `No` (the number column of your file); `File Order` is added only
when the file has no number column.

## Run locally
1. Python 3.10+. Install the **same** scikit-learn / xgboost / lightgbm versions the models were trained with
   (the Settings page tells you the scikit-learn version, see below).
2. `pip install -r deployment_app/requirements.txt`
3. From the repository root: `streamlit run deployment_app/app.py`
4. Put the `.pkl` files in `deployment_app/checkpoints/`, or upload them on the Settings page.

## Deploy on Streamlit Community Cloud (permanent link)
**1. Repository layout** (root of the GitHub repository):
```
<your-repo>/
├── .streamlit/config.toml        ← must be at the ROOT when the app is in a subfolder
└── deployment_app/
    ├── app.py                    ← "Main file path": deployment_app/app.py
    ├── requirements.txt
    ├── remote_checkpoints.py
    ├── checkpoints/              ← small checkpoints (< 100 MB) can go here
    └── app_pages/ ...
```
The GitHub web page cannot upload hidden folders reliably: for `.streamlit/config.toml` use
**Add file → Create new file**, type `.streamlit/config.toml` as the name and paste the content. The web uploader
also refuses files above 25 MB; use `git` or GitHub Desktop for files up to 100 MB.

**2. Checkpoints.** Under 100 MB: `git push` them into `deployment_app/checkpoints/`. Larger: see "Large
checkpoints" below.

**3. Deploy.** share.streamlit.io → **Create app** → Repository `<user>/<repo>`, Branch `main`, Main file path
`deployment_app/app.py`. Under **Advanced settings** choose the Python version (default 3.12) and, for a private
repository with large checkpoints, the Secrets (below). Then **Deploy**.

**4. Check the versions.** Open **Settings** in the running app. If the checkpoints were trained with another
scikit-learn version than the server runs, a yellow box gives the exact line to add to
`deployment_app/requirements.txt`, e.g. `scikit-learn==1.5.2`. Push it; the app redeploys. If pip cannot install
that version for the server's Python, deploy again with an older Python. xgboost / lightgbm / catboost versions
are not stored in the file: pin the versions you trained with if loading fails.

**5. Updates.** Every `git push` to the branch redeploys automatically. After changing Secrets:
⋮ → **Reboot app**. The Python version cannot be changed after deployment: delete the app and deploy again
(the subdomain becomes free immediately, so the link can stay the same; re-enter the Secrets).

### Large checkpoints (> 100 MB)
GitHub refuses files above 100 MB in a normal push, but a **Release** accepts files up to 2 GB.
1. Repository → **Releases** → **Create a new release** (tag e.g. `checkpoints-v1`) → attach the `.pkl` files →
   **Publish release**.
2. Right-click a file on the release page → **Copy link address**.
3. Put one entry per file in `deployment_app/remote_checkpoints.py`:
   ```python
   REMOTE_CHECKPOINTS = [
       {"filename": "checkpoint_reg_No-Aug_Optuna.pkl",
        "url": "https://github.com/<user>/<repo>/releases/download/checkpoints-v1/checkpoint_reg_No-Aug_Optuna.pkl"},
   ]
   ```
4. The app downloads each file at start-up (streamed to disk, size-checked, skipped when already present) and
   loads it like any other checkpoint. Hosting platforms may erase the disk when an app restarts; the files are
   then downloaded again.

**Private repository:** the release files are then only downloadable with a token. GitHub → Settings → Developer
settings → Personal access tokens → **Fine-grained tokens** → Generate new token → Repository access: only this
repository → Permissions: **Contents: Read-only**. Paste it into the app's **Secrets** (Advanced settings when
deploying, or ⋮ → Settings → Secrets):
```toml
GITHUB_TOKEN = "github_pat_..."
```
The token must never be written in a file of the repository.

Memory: one shared copy of each checkpoint is kept for all visitors, and a checkpoint is read straight from its
file. Community Cloud gives roughly 2.7 GB of memory per app.

## Change log
### Version 11
- The app no longer contains anything platform-specific: extra checkpoint folders come from the generic
  `CHECKPOINT_DIRS` variable.
- Settings shows the scikit-learn version each checkpoint was trained with (read from the file, without
  unpickling it) next to the running one, with the exact `requirements.txt` line to use.

### Version 10
- Large checkpoints via GitHub Releases (`remote_checkpoints.py`), private repositories via `GITHUB_TOKEN`.
- One shared, stream-loaded copy of each checkpoint for all visitors.
- `.streamlit/config.toml` also at the repository root; absolute page paths in `app.py`.

### Version 9
- Two numbering columns only (`#` and `No`).

### Version 8
- Result tables: identity, prediction, coloured confidence and its reason first; parameters last. Pinned
  columns, frozen Excel panes, responsive Vega-Lite charts.
- Fixed multi-class SHAP values (they cancelled out) and page loading after a browser refresh (`app_pages/`).

### Version 5–7
- Manual upload overrides automatic checkpoints; model/scenario selection with metrics; CR confidence; editable
  schema; batch templates; xlsx downloads; English interface; fixed a batch-upload key collision.
