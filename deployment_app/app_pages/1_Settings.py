import pandas as pd
import streamlit as st

from model_utils import (activate_clf, activate_reg, active_clf, active_reg, auto_load_once,
                         candidate_label, clf_source_from, discover_checkpoint_candidates,
                         extract_schema, get_categories_from_checkpoint, get_numeric_ranges,
                         get_training_defaults, load_candidate, load_checkpoint_verbose,
                         metrics_row, model_metrics_table, model_names, quality_label,
                         reg_scenario_cr_type, resolve_display, scenario_summary,
                         set_reg_scenario, train_score)

init = st.session_state
st.title("⚙️ Settings")
st.caption("Choose which checkpoints and models are used, check their quality, and adjust the input "
           "schema. Everything here applies to all prediction pages.")
auto_load_once(st)  # no-op if app.py already ran it

if init["_auto_discover_msgs"]:
    st.success("🔎 Checkpoints detected automatically:  \n" + "  \n".join(init["_auto_discover_msgs"]))


def show_load_warnings(msgs):
    if msgs:
        st.warning("⚠️ Library-version warnings while loading the checkpoint — the scikit-learn / xgboost / "
                   "lightgbm versions here differ from the training environment. Predictions may fail or "
                   "differ from the notebook; install the same versions (see README).  \n- "
                   + "  \n- ".join(msgs))


show_load_warnings(init.get("_load_warnings"))

# ======================================================================
# 1. Checkpoints found automatically
# ======================================================================
st.header("1. Checkpoints found automatically")
st.caption("The app scans `/kaggle/input` (Experiment 1/2 notebook Outputs attached with "
           "**+ Add Input → Notebooks**, or Datasets) and the local `checkpoints/` folder for "
           "`checkpoint_reg_*.pkl` and `checkpoint_p2_*.pkl`. No need to separate the .pkl files.")
if st.button("🔄 Rescan"):
    init["_ckpt_candidates"] = discover_checkpoint_candidates()
    st.rerun()

cands = init["_ckpt_candidates"]
reg_cands = [c for c in cands if c["kind"] == "regression"]
clf_cands = [c for c in cands if c["kind"] == "classification"]
if not cands:
    st.info("No checkpoint found yet. On Kaggle: attach the Experiment 1 & 2 notebook Outputs "
            "(**+ Add Input → Notebooks**) and click **Rescan**. Locally: put the .pkl files in "
            "`checkpoints/` next to `app.py`, or upload them manually in section 2.")
else:
    st.dataframe(pd.DataFrame([{
        "Type": "Corrosion rate (Exp. 1)" if c["kind"] == "regression" else "Risk (Exp. 2)",
        "File": c["name"], "TAG": c["tag"], "CR type": c["source"] or "-",
        "Size (MB)": round(c["size"] / 1e6, 1), "Location": c["path"],
        "Active": "✅" if c["path"] in (init["reg_path"], init["clf_path"]) else "",
    } for c in cands]), hide_index=True, width="stretch")

    k1, k2 = st.columns(2)
    for col, lst, kind, cur_key in ((k1, reg_cands, "regression", "reg_path"),
                                     (k2, clf_cands, "classification", "clf_path")):
        with col:
            if not lst:
                st.caption(f"No {kind} checkpoint found.")
                continue
            paths = [c["path"] for c in lst]
            cur = init[cur_key]
            i = st.selectbox(f"{'Corrosion-rate' if kind == 'regression' else 'Risk'} checkpoint",
                             range(len(lst)), index=paths.index(cur) if cur in paths else 0,
                             format_func=lambda j, lst=lst: candidate_label(lst[j]), key=f"pick_{kind}")
            if st.button("Load this checkpoint", key=f"load_pick_{kind}", disabled=lst[i]["path"] == cur):
                with st.spinner("Loading checkpoint..."):
                    try:
                        k, obj, w = load_candidate(lst[i])
                        if k != kind:
                            st.error(f"This file is not a {kind} checkpoint.")
                        else:
                            if kind == "regression":
                                activate_reg(st, obj, lst[i]["path"], lst[i]["name"])
                            else:
                                activate_clf(st, obj, lst[i]["path"], lst[i]["name"], lst[i]["source"])
                            init["_load_warnings"] = w
                            st.rerun()
                    except Exception as e:
                        st.error(f"Could not load: {e}")

st.divider()

# ======================================================================
# 2. Manual upload (always overrides the automatic choice)
# ======================================================================
st.header("2. Upload a checkpoint manually (optional)")
st.caption("An uploaded file **replaces** the automatically loaded checkpoint of the same type. The "
           "input features of each model are read automatically from its own .pkl file, so checkpoints "
           "trained with a different number of features work without changing anything.")


def new_upload(uploaded, slot):
    """Process each upload ONCE (a file stays in st.file_uploader across reruns)."""
    if uploaded is None:
        return False
    fid = getattr(uploaded, "file_id", None) or f"{uploaded.name}-{uploaded.size}"
    if init.get(f"_last_upload_{slot}") == fid:
        return False
    init[f"_last_upload_{slot}"] = fid
    return True


u1, u2 = st.columns(2)
with u1:
    up_reg = st.file_uploader("Corrosion-rate checkpoint (checkpoint_reg_*.pkl)", type=["pkl"], key="up_reg")
    if new_upload(up_reg, "reg"):
        try:
            with st.spinner("Loading checkpoint..."):
                kind, obj, w = load_checkpoint_verbose(up_reg.getvalue())
            if kind != "regression":
                st.error("This file is a RISK (classification) checkpoint — upload it on the right.")
            else:
                activate_reg(st, obj, None, up_reg.name)
                init["_load_warnings"] = w
                n, c = extract_schema(obj[init["reg_scenario"]])
                st.success(f"Loaded {up_reg.name}: {len(obj)} scenario(s), {len(n)} numeric + {len(c)} "
                           f"categorical features detected.")
                show_load_warnings(w)
        except Exception as e:
            st.error(f"Could not load: {e}")
with u2:
    up_clf = st.file_uploader("Risk checkpoint (checkpoint_p2_*.pkl)", type=["pkl"], key="up_clf")
    if new_upload(up_clf, "clf"):
        try:
            with st.spinner("Loading checkpoint..."):
                kind, obj, w = load_checkpoint_verbose(up_clf.getvalue())
            if kind != "classification":
                st.error("This file is a CORROSION-RATE (regression) checkpoint — upload it on the left.")
            else:
                activate_clf(st, obj, None, up_clf.name, clf_source_from(obj, up_clf.name))
                init["_load_warnings"] = w
                n, c = extract_schema(obj)
                st.success(f"Loaded {up_clf.name}: {len(n)} numeric + {len(c)} categorical features detected.")
                show_load_warnings(w)
        except Exception as e:
            st.error(f"Could not load: {e}")

st.divider()

# ======================================================================
# 3. Model selection & quality
# ======================================================================
st.header("3. Model selection & quality")
st.caption("By default the best scenario (lowest cross-validated RMSE, matching the risk model's CR type) "
           "and its best model are used. You can choose any scenario and model manually. Quality labels "
           "are a rule of thumb: corrosion rate R²(CV) ≥ 0.75 Good, ≥ 0.50 Fair, else Weak; risk "
           "F1-macro(CV) ≥ 0.75 Good, ≥ 0.60 Fair, else Weak.")
QUAL_ICON = {"Good": "🟢", "Fair": "🟡", "Weak": "🔴", "Unknown": "⚪"}


def fmt_table(df):
    num = [c for c in df.columns if df[c].dtype.kind == "f"]
    return df.style.format({c: "{:.4f}" for c in num}, na_rep="")


tab_r, tab_c = st.tabs(["📈 Corrosion rate (Experiment 1)", "⚠️ Risk (Experiment 2)"])
with tab_r:
    reg_res, reg_model = active_reg(st)
    if reg_res is None:
        st.info("No corrosion-rate checkpoint loaded (section 1 or 2).")
    else:
        store = init["reg_store"]
        st.caption(f"Checkpoint: `{init['reg_path'] or init['reg_label'] or 'manual upload'}`")
        st.markdown("**Best model of every scenario**")
        st.dataframe(fmt_table(scenario_summary(store)), hide_index=True, width="stretch")
        keys = list(store.keys())
        s1, s2 = st.columns(2)
        with s1:
            sc = st.selectbox("Scenario (CR type × target transform)", keys,
                              index=keys.index(init["reg_scenario"]),
                              format_func=lambda k: f"{k}  (best: {store[k]['best_name']})")
            if sc != init["reg_scenario"]:
                set_reg_scenario(st, sc)
                st.rerun()
        res = store[init["reg_scenario"]]
        names = model_names(res)
        with s2:
            def lab(m):
                r = metrics_row(res, m) or {}
                v = r.get("R2_CV", r.get("R2"))
                return f"{m}{' ★ best' if m == res['best_name'] else ''}" + (f"  (R² CV {v:.3f})" if v is not None and pd.notna(v) else "")
            m = st.selectbox("Model", names, index=names.index(reg_model), format_func=lab, key="sel_reg_model")
            init["reg_model"] = m
        q, note = quality_label(metrics_row(res, m), "regression", train_score(res, m, "regression"))
        st.markdown(f"**Selected:** {m} — quality {QUAL_ICON[q]} **{q}** ({note})")
        st.markdown(f"**All models in scenario {init['reg_scenario']}**")
        st.dataframe(fmt_table(model_metrics_table(res, "regression")), hide_index=True, width="stretch")
        if res.get("_pre_reconstructed"):
            st.caption("ℹ️ This checkpoint does not store its preprocessor — it was rebuilt exactly from the "
                       "stored training matrix (no retraining).")
        if st.button("🗑️ Unload corrosion-rate model", key="clear_reg"):
            for k in ("reg_store", "reg_scenario", "reg_model", "reg_path", "reg_label"):
                init[k] = None
            st.rerun()

with tab_c:
    clf_res, clf_model = active_clf(st)
    if clf_res is None:
        st.info("No risk checkpoint loaded (section 1 or 2).")
    else:
        st.caption(f"Checkpoint: `{init['clf_path'] or init['clf_label'] or 'manual upload'}`"
                   + (f" | trained with CR type **{init['clf_source']}**" if init["clf_source"] else ""))
        names = model_names(clf_res)

        def labc(m):
            r = metrics_row(clf_res, m) or {}
            v = r.get("F1_macro_CV", r.get("F1_macro"))
            return f"{m}{' ★ best' if m == clf_res['best_name'] else ''}" + (f"  (F1-macro CV {v:.3f})" if v is not None and pd.notna(v) else "")
        m = st.selectbox("Model", names, index=names.index(clf_model), format_func=labc, key="sel_clf_model")
        init["clf_model"] = m
        q, note = quality_label(metrics_row(clf_res, m), "classification", train_score(clf_res, m, "classification"))
        st.markdown(f"**Selected:** {m} — quality {QUAL_ICON[q]} **{q}** ({note})")
        st.markdown("**All models in this checkpoint**")
        st.dataframe(fmt_table(model_metrics_table(clf_res, "classification")), hide_index=True, width="stretch")
        if st.button("🗑️ Unload risk model", key="clear_clf"):
            for k in ("clf_res", "clf_model", "clf_path", "clf_label", "clf_source"):
                init[k] = None
            st.rerun()

if init["reg_store"] and init["clf_res"] and init["clf_source"]:
    rt = reg_scenario_cr_type(init["reg_scenario"])
    if rt != init["clf_source"]:
        st.warning(f"Scenario **{init['reg_scenario']}** predicts *{rt}* CR, but the risk model was trained "
                   f"with *{init['clf_source']}* CR. For the Combined page choose a "
                   f"{init['clf_source'].capitalize()}_* scenario.")

st.divider()

# ======================================================================
# 4. Input schema (auto-detected, editable)
# ======================================================================
st.header("4. Input schema (auto-detected, editable)")
st.caption("Columns are read automatically from each loaded model. You can change how each column is "
           "shown and entered: **Display label** (forms and results), **Excel header** (batch template "
           "and upload — the display label and internal name are also accepted), **Default value** (pre-"
           "fills the form and fills empty/hidden cells; blank = training median / most frequent value), "
           "**Extra categories** (additional choices in dropdowns) and **Show in form** (hidden columns "
           "use their default). Settings are shared by columns with the same name in both models.")


def schema_editor(res, title):
    try:
        num_cols, cat_cols = extract_schema(res)
    except Exception as e:
        st.error(f"Could not read the schema: {e}")
        return
    st.markdown(f"**{title}** — {len(num_cols)} numeric and {len(cat_cols)} categorical columns detected.")
    ranges = get_numeric_ranges(res, num_cols)
    cats_auto = get_categories_from_checkpoint(res, cat_cols)
    train_def = get_training_defaults(res)
    rows = []
    for c in num_cols + cat_cols:
        is_num = c in num_cols
        label = resolve_display(c, init["display_overrides"])
        if is_num:
            lo, hi = ranges.get(c, (None, None))
            info = f"{lo:.4g} to {hi:.4g}" if lo is not None else "(not available)"
        else:
            info = ", ".join(cats_auto.get(c, [])) or "(not available)"
        td = train_def.get(c)
        rows.append({
            "Column": c, "Type": "Numeric" if is_num else "Categorical",
            "Display label": label,
            "Excel header": init["header_overrides"].get(c) or label,
            "Default value": str(init["default_overrides"].get(c, "")),
            "Training default": (f"{td:.4g}" if is_num and td is not None else str(td if td is not None else "")),
            "Extra categories (comma-separated)": ", ".join(init["category_overrides"].get(c, [])) if not is_num else "",
            "Training range / categories": info,
            "Show in form": True if c == "cr" else (c not in init["excluded_cols"]),
        })
    edited = st.data_editor(
        pd.DataFrame(rows), key=f"editor_{title}", hide_index=True, width="stretch",
        disabled=["Column", "Type", "Training default", "Training range / categories"],
        column_config={"Show in form": st.column_config.CheckboxColumn(),
                       "Default value": st.column_config.TextColumn(help="Leave blank to use the training default")},
    )
    if "cr" in num_cols:
        st.caption("**cr** (corrosion rate) is always an input on the Risk page and is predicted on the "
                   "Combined page, so it cannot be hidden.")
    if st.button(f"💾 Save {title.lower()} schema", key=f"save_{title}"):
        errors = []
        for _, r in edited.iterrows():
            c = r["Column"]
            label = (r["Display label"] or "").strip() or resolve_display(c, {})
            init["display_overrides"][c] = label
            header = (r["Excel header"] or "").strip()
            if header and header != label:
                init["header_overrides"][c] = header
            else:
                init["header_overrides"].pop(c, None)
            dv = str(r["Default value"] or "").strip()
            if dv == "":
                init["default_overrides"].pop(c, None)
            elif r["Type"] == "Numeric":
                try:
                    init["default_overrides"][c] = float(dv)
                except ValueError:
                    errors.append(f"{c}: '{dv}' is not a number")
            else:
                init["default_overrides"][c] = dv
            if r["Type"] == "Categorical":
                extra = [v.strip() for v in str(r["Extra categories (comma-separated)"] or "").split(",") if v.strip()]
                if extra:
                    init["category_overrides"][c] = extra
                else:
                    init["category_overrides"].pop(c, None)
            if c != "cr":
                (init["excluded_cols"].discard if r["Show in form"] else init["excluded_cols"].add)(c)
        if errors:
            st.error("Not saved for: " + "; ".join(errors))
        else:
            st.success("Schema saved.")


t1, t2 = st.tabs(["Corrosion-rate model schema", "Risk model schema"])
with t1:
    r, _ = active_reg(st)
    if r is not None:
        schema_editor(r, "Corrosion-rate")
    else:
        st.info("Load a corrosion-rate checkpoint first.")
with t2:
    r, _ = active_clf(st)
    if r is not None:
        schema_editor(r, "Risk")
    else:
        st.info("Load a risk checkpoint first.")

if st.button("↩️ Reset all schema changes (labels, headers, defaults, extra categories, hidden columns)"):
    for k in ("display_overrides", "header_overrides", "default_overrides", "category_overrides"):
        init[k] = {}
    init["excluded_cols"] = set()
    for k in list(init.keys()):
        if str(k).startswith("editor_"):
            del init[k]
    st.rerun()

st.caption("Next: open **Corrosion Rate Prediction**, **Risk Prediction** or **Combined** in the menu.")
