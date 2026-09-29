import pandas as pd
import streamlit as st

from info_utils import show_limitations
from model_utils import COL_RISK, active_clf, extract_schema, predict_risk
from shap_utils import show_shap_batch, show_shap_single
from ui_utils import (add_history, arrange_results, build_specs, confidence_legend, download_xlsx,
                      file_key, history_section, input_form, mode_selector, model_info_items,
                      model_info_sheet, read_batch, row_ref_note, render_risk_summary_cards, risk_bar, risk_block,
                      risk_distribution_table, row_from_form, show_results_table, show_risk_result,
                      template_button, to_display_columns)

ss = st.session_state
st.title("⚠️ Risk Prediction (Experiment 2)")
res, model = active_clf(st)
if res is None:
    st.warning("No risk model loaded. Open **⚙️ Settings** in the menu first.")
    st.stop()

overrides = ss["display_overrides"]
specs = build_specs([res], include_cr=True)
num_all, _ = extract_schema(res)
st.caption(f"Active model: **{model}** | Hyperparameters: {res.get('overall_source', '?')}"
           + (f" | trained with CR type {ss['clf_source']}" if ss.get("clf_source") else "")
           + " — change in ⚙️ Settings")
show_limitations(res, model, [c for c in num_all if c != "cr"], "classification", overrides)
info_sheet = model_info_sheet(model_info_items("Risk", res, model, ss.get("clf_label")))
sig = (ss.get("clf_label"), model)
model_col = pd.DataFrame({"Risk Model": [model]})


def predict(df, file_order=None):
    out, Xt = predict_risk(res, df, model)
    table = arrange_results(to_display_columns(df, specs), specs, [risk_block(out)],
                            file_order=file_order, sort="risk" if file_order is not None else None,
                            extra_tail=pd.concat([model_col] * len(df), ignore_index=True))
    return out, Xt, table


if mode_selector("risk_mode").startswith("Manual"):
    st.subheader("Equipment data")
    vals = input_form(specs, "risk")
    if st.button("🔮 Predict risk", type="primary", key="btn_risk"):
        df_in = row_from_form(specs, vals)
        out, Xt, table = predict(df_in)
        ss["last_risk"] = dict(sig=sig, row=out.iloc[0].to_dict(), Xt=Xt, result=table)
        add_history("hist_risk", table)

    last = ss.get("last_risk")
    if last:
        if last["sig"] != sig:
            st.info("The result below was computed with a different model; press Predict again to update it.")
        st.divider()
        confidence_legend("risk")
        st.write("")
        show_risk_result(last["row"])
        download_xlsx("⬇️ Download this result (.xlsx)", {"Result": last["result"], "Model info": info_sheet},
                      "risk_prediction.xlsx", key="dl_risk_single", primary=True)
        st.subheader("Explanation (SHAP)")
        show_shap_single(res, last["Xt"], model)
    history_section("hist_risk", "risk_history.xlsx", {"Model info": info_sheet})
else:
    st.subheader("Batch upload")
    st.markdown("Download the template → fill one row per equipment in sheet **Input** (including the "
                "corrosion rate) → upload it.")
    template_button(specs, "template_risk.xlsx", "tmpl_risk")
    f = st.file_uploader("Upload the filled file (.xlsx / .csv)", type=["xlsx", "xls", "csv"], key="upload_batch_risk")
    if f is not None:
        cache_key = (file_key(f), sig, str(ss["default_overrides"]), str(ss["header_overrides"]), str(overrides))
        if ss.get("_cache_batch_risk", {}).get("key") != cache_key:
            with st.spinner("Predicting..."):
                df = read_batch(f, specs)
                out, Xt, table = predict(df, file_order=df.attrs["file_order"])
                ss["_cache_batch_risk"] = dict(key=cache_key, table=table, Xt=Xt)
        b = ss["_cache_batch_risk"]
        t = b["table"]
        st.success(f"{len(t)} rows predicted — sorted from the highest risk ({row_ref_note(t)}).")
        render_risk_summary_cards(t[COL_RISK])
        dist = risk_distribution_table(t)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Risk distribution**")
            st.dataframe(dist, hide_index=True, width="stretch")
        with c2:
            risk_bar(t)
        confidence_legend("risk")
        st.write("")
        show_results_table(t, height=460)
        download_xlsx("⬇️ Download results (.xlsx)",
                      {"Results": t, "Risk distribution": dist, "Model info": info_sheet},
                      "risk_batch_results.xlsx", key="dl_risk_batch", primary=True)
        st.subheader("Explanation (SHAP, all rows)")
        show_shap_batch(res, b["Xt"], model)
