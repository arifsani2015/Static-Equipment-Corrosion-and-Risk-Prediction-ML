# Corrosion Rate & Risk Prediction — Deployment App

Streamlit app for the models of **Experiment 1** (corrosion-rate regression) and
**Experiment 2** (API RP 581 risk classification). It runs locally, on Kaggle
(integrated with the notebook Outputs) or on Streamlit Community Cloud.

## Pages
(Page files live in `app_pages/`.)

| Page | What it does |
|---|---|
| ⚙️ **Settings** | Checkpoints (auto-detected or manual upload), scenario & model selection with quality metrics, editable input schema |
| 📈 **Corrosion Rate Prediction** | Predicts CR with an 80% interval, a confidence level and the 5 most similar real equipment |
| ⚠️ **Risk Prediction** | Predicts the risk class and shows the probability of **every** class |
| 🔗 **Combined (CR → Risk)** | For equipment without CR: predicts CR first, then risk (same flow as Experiment 3) |

Every prediction page supports **manual input** (one equipment) and **batch upload**
(downloadable Excel template), and every result — single, history or batch — can be
downloaded as **.xlsx**.

## Checkpoints and models
- **Automatic detection**: the app scans `/kaggle/input` (any depth — notebook Outputs
  or Datasets) and the local `checkpoints/` folder for `checkpoint_reg_{TAG}.pkl` and
  `checkpoint_p2_{crtype}_{TAG}.pkl`. Default pair: same TAG, risk model trained with
  *selected* CR.
- **Manual upload** on the Settings page always **replaces** the automatically loaded
  checkpoint of the same type.
- **Features are read from each .pkl** (from the fitted preprocessor), so checkpoints
  trained with a different number or set of features work without code changes.
- **Default model**: the scenario with the lowest cross-validated RMSE (preferring the
  CR type of the risk model) and its best model. Any scenario (e.g. log1p vs raw) and
  any model can be chosen manually.
- **Quality labels** (rule of thumb): corrosion rate R²(CV) ≥ 0.75 Good, ≥ 0.50 Fair,
  else Weak; risk F1-macro(CV) ≥ 0.75 Good, ≥ 0.60 Fair, else Weak. A note is added when
  the training score exceeds the test score by more than 0.20 (possible overfitting).

## Corrosion-rate confidence
Each CR prediction comes with:
1. **80% prediction interval** — built from the chosen model's errors on the test set
   (log scale, conditional on prediction size when enough data). Conservative
   split-conformal quantiles are used so that at least 80% of test values fall inside.
2. **Model agreement** — spread (std/mean) of the predictions of all models in the scenario.
3. **Similarity to known equipment** — distance to the 5 nearest real equipment, compared
   with typical distances between real equipment: Inside (≤ 90th percentile), Borderline
   (≤ 99th) or Outside (extrapolation).
4. **Similar equipment** — the 5 most similar real equipment with their identity and
   **measured CR**, and whether the prediction lies within their range.

**Confidence level**: Outside → Low. Otherwise +1 Inside, +1 agreement ≤ 15% (−1 if
> 35%), +1 within the range of similar equipment; ≥ 2 High, ≥ 0 Medium, else Low.
Indicators that cannot be computed for a checkpoint (e.g. no test set stored) are shown
as "n/a".

## Input schema (Settings → section 4)
Auto-detected from the checkpoint, then editable per column: display label, Excel header
(template/batch; the label and internal name are also accepted), default value (form
pre-fill and empty/hidden cells; blank = training median / most frequent value), extra
categories, and show/hide in forms. `cr` cannot be hidden.

## Batch template
Sheets: **Input** (fill this), **Column guide** (meaning, type, allowed values/range,
default), **Example** and **Instructions**. Extra columns such as Tag No are kept in the
results. Empty cells are filled with the default.

## Run on Kaggle
Use `Jalankan_di_Kaggle.ipynb` (one folder above `deployment_app/`):
1. **File → Import Notebook** on Kaggle.
2. **+ Add Input → Notebooks** → add the Experiment 1 and Experiment 2 notebooks (their
   latest *committed* version — Save & Run All).
3. **Settings → Internet: On**, then **Run All**. Cell C2 prints the link
   (`https://….trycloudflare.com`). Cell C3 only stops the app when `STOP_APP = True`.

## Run locally
1. Python 3.10+. Install the **same** scikit-learn / xgboost / lightgbm versions as the
   Kaggle training environment (check with `print(sklearn.__version__)` etc. on Kaggle);
   otherwise the Settings page shows a version warning.
2. `pip install -r requirements.txt`, then `streamlit run app.py` (works from any folder).
3. Put the `.pkl` files in `deployment_app/checkpoints/` for automatic loading, or upload
   them on the Settings page.

## Run online (Streamlit Community Cloud)
Push this folder to GitHub → https://streamlit.io/cloud → New app → `app.py`. Upload the
checkpoints on the Settings page each session.

## Change log
### Version 9
- Only two numbering columns: `#` (rank in the displayed table) and `No` (the row number from your
  own file, e.g. `No`, `Tag No`, `ID`). `File Order` (row position in the uploaded file) is added
  only when the file has no number column at all, so a result can always be traced back to its row.

### Version 8
- Result tables (screen and Excel): `#` (row number), then the equipment identity (No, Equipment/Pipe Name, Equipment Part...), then the
  prediction, its confidence (coloured) and the reason, then details, then the input parameters.
  Row numbers and equipment name stay pinned while scrolling; Excel panes are frozen the same way.
- Confidence levels explained above every result; single results show prediction | confidence |
  reason side by side. Risk confidence = probability of the predicted class (≥70% High,
  50–70% Medium, <50% Low) with the runner-up and margin as the note.
- Manual forms have optional Equipment/Pipe Name and Equipment Part fields (kept in history).
- Charts rebuilt with Altair/Vega-Lite: proportional and sharp at any width or zoom level;
  responsive font sizes in the result cards.
- Fixed: multi-class SHAP values were averaged over classes BEFORE taking absolute values, so they
  cancelled out (~1e-16). Batch importance now equals notebook D6; single explanations show the
  contribution towards the predicted class.
- Fixed: refreshing / opening a prediction page directly skipped app.py (checkpoints not loaded);
  pages moved from `pages/` to `app_pages/` so `st.navigation` always runs app.py first.

### Version 7
- Fixed: batch upload crashed on all three prediction pages ("'UploadedFile' object has no
  attribute 'get'"). The uploader widget key was the same as the result-cache key, so
  Streamlit stored the uploaded file where the cache was expected. Uploader keys are now
  `upload_batch_*` and caches `_cache_batch_*`; all widget/state keys audited — no other collisions.

### Version 6
- Fixed: "most similar equipment" crashed when an identity cell (e.g. `No`) is empty (pandas >= 3).

### Version 5
- Whole interface, Kaggle notebook and README in English.
- Manual upload clearly overrides auto-detected checkpoints; features read per checkpoint.
- Scenario and model selection (default = best), metrics table of every model, best model
  per scenario, quality labels and overfitting notes.
- Risk results show the probability of every class as numbers, the runner-up class and
  the margin; batch results add a risk-distribution table (counts, shares, mean
  probabilities, expected counts).
- Corrosion-rate confidence: 80% interval, model agreement, applicability domain,
  similar real equipment, High/Medium/Low level.
- Schema editor extended with Excel header and default value.
- Batch template with column guide/example/instructions; all results (single, history,
  batch) downloadable as .xlsx; manual results persist across interactions; forms are
  pre-filled with training medians / most frequent values.

### Version 4
- Experiment 1 checkpoints did not store the preprocessor → rebuilt exactly from Xtr_t
  (predictions identical); Experiment 1 notebook now also stores it.
- `phase` means different Excel columns in Experiments 1 and 2 → separate inputs
  (Majority Fluid Phase / Fluid Release Phase).
- Training ranges never displayed; Streamlit version pin (≥ 1.49); scenario selection
  reset after manual upload; tunnel upload 403; config missing on Kaggle; search folders
  relative to app.py; Kaggle C3 cell stopping the app during Run All.
