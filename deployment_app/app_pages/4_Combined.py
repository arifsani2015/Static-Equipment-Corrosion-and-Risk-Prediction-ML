import pandas as pd
import streamlit as st

from info_utils import show_limitations
from model_utils import (COL_CR, COL_CR_CONF, COL_RISK, active_clf, active_reg, cr_reliability,
                         extract_schema, predict_cr, predict_risk, reg_scenario_cr_type,
                         reliability_basis, similar_equipment)
from shap_utils import show_shap_batch, show_shap_single
from ui_utils import (add_history, arrange_results, build_specs, confidence_legend,
                      confidence_summary_cards, cr_block, cr_histogram, download_xlsx, file_key,
                      history_section, input_form, mode_selector, model_info_items, model_info_sheet,
                      read_batch, row_ref_note, render_risk_summary_cards, risk_bar, risk_block,
                      risk_distribution_table, row_from_form, show_cr_result, show_results_table,
                      show_risk_result, template_button, to_display_columns)

ss = st.session_state
st.title("🔗 Combined: Corrosion Rate → Risk")
st.caption("For equipment WITHOUT a corrosion rate: the corrosion rate is predicted first (Experiment 1) "
           "and then used as input for the risk prediction (Experiment 2) — the same flow as Experiment 3.")
reg_res, reg_model = active_reg(st)
clf_res, clf_model = active_clf(st)
if reg_res is None or clf_res is None:
    st.warning("Both models are needed. Open **⚙️ Settings** in the menu first.")
    st.stop()

overrides = ss["display_overrides"]
specs = build_specs([reg_res, clf_res], include_cr=False)
num_reg, cat_reg = extract_schema(reg_res)
num_clf, _ = extract_schema(clf_res)
st.caption(f"Corrosion rate: **{reg_model}** ({ss['reg_scenario']}) | Risk: **{clf_model}** — change in ⚙️ Settings")
rt, ct = reg_scenario_cr_type(ss["reg_scenario"]), ss.get("clf_source")
if ct and rt != ct:
    st.warning(f"Scenario **{ss['reg_scenario']}** predicts *{rt}* CR, but the risk model was trained with "
               f"*{ct}* CR. Results are computed but are not consistent with Experiment 3 — change the "
               f"scenario in ⚙️ Settings.")
t1, t2 = st.tabs(["Corrosion-rate model limitations", "Risk model limitations"])
with t1:
    show_limitations(reg_res, reg_model, num_reg, "regression", overrides)
with t2:
    show_limitations(clf_res, clf_model, [c for c in num_clf if c != "cr"], "classification", overrides)
info_sheet = model_info_sheet(
    model_info_items("CR", reg_res, reg_model, ss.get("reg_label"), ss["reg_scenario"])
    + model_info_items("Risk", clf_res, clf_model, ss.get("clf_label")))
sig = (ss.get("reg_label"), ss["reg_scenario"], reg_model, ss.get("clf_label"), clf_model)


def run(df, file_order=None):
    out_cr, Xt_cr = predict_cr(reg_res, df[num_reg + cat_reg], reg_model)
    rel = cr_reliability(reg_res, Xt_cr, out_cr["Predicted_CR"].values, reg_model)
    df_risk = df.copy()
    df_risk["cr"] = out_cr["Predicted_CR"].values
    out_risk, Xt_risk = predict_risk(clf_res, df_risk, clf_model)
    models = pd.DataFrame({"CR Model": [reg_model] * len(df), "Risk Model": [clf_model] * len(df)})
    table = arrange_results(to_display_columns(df, specs), specs,
                            [cr_block(out_cr["Predicted_CR"].values, rel), risk_block(out_risk)],
                            file_order=file_order, sort="risk" if file_order is not None else None,
                            extra_tail=models)
    return out_cr, Xt_cr, rel, out_risk, Xt_risk, table


if mode_selector("combo_mode").startswith("Manual"):
    st.subheader("Equipment data (combined schema of both models)")
    vals = input_form(specs, "combo")
    if st.button("🔮 Predict corrosion rate, then risk", type="primary", key="btn_combo"):
        df_in = row_from_form(specs, vals)
        out_cr, Xt_cr, rel, out_risk, Xt_risk, table = run(df_in)
        ss["last_combo"] = dict(sig=sig, pred=float(out_cr["Predicted_CR"].iloc[0]), rel=rel.iloc[0].to_dict(),
                                sim=similar_equipment(reg_res, Xt_cr, reg_model, overrides),
                                row=out_risk.iloc[0].to_dict(), Xt=Xt_risk, result=table)
        add_history("hist_combo", table)

    last = ss.get("last_combo")
    if last:
        if last["sig"] != sig:
            st.info("The result below was computed with different models; press Predict again to update it.")
        st.divider()
        confidence_legend("combo")
        st.subheader("1️⃣ Corrosion rate")
        show_cr_result(last["pred"], last["rel"], last["sim"], reliability_basis(reg_res, reg_model))
        st.subheader("2️⃣ Risk (using the predicted corrosion rate)")
        show_risk_result(last["row"])
        download_xlsx("⬇️ Download this result (.xlsx)",
                      {"Result": last["result"], "Similar equipment": last["sim"], "Model info": info_sheet},
                      "combined_prediction.xlsx", key="dl_combo_single", primary=True)
        st.subheader("Explanation of the risk prediction (SHAP)")
        show_shap_single(clf_res, last["Xt"], clf_model)
    history_section("hist_combo", "combined_history.xlsx", {"Model info": info_sheet})
else:
    st.subheader("Batch upload (equipment WITHOUT corrosion rate)")
    st.markdown("Download the template → fill one row per equipment in sheet **Input** → upload it.")
    template_button(specs, "template_combined.xlsx", "tmpl_combo")
    f = st.file_uploader("Upload the filled file (.xlsx / .csv)", type=["xlsx", "xls", "csv"], key="upload_batch_combo")
    if f is not None:
        cache_key = (file_key(f), sig, str(ss["default_overrides"]), str(ss["header_overrides"]), str(overrides))
        if ss.get("_cache_batch_combo", {}).get("key") != cache_key:
            with st.spinner("Predicting corrosion rate, then risk..."):
                df = read_batch(f, specs)
                out_cr, _Xt_cr, rel, out_risk, Xt_risk, table = run(df, file_order=df.attrs["file_order"])
                ss["_cache_batch_combo"] = dict(key=cache_key, table=table, Xt=Xt_risk)
        b = ss["_cache_batch_combo"]
        t = b["table"]
        st.success(f"{len(t)} rows predicted (corrosion rate, then risk) — sorted from the highest risk "
                   f"({row_ref_note(t)}).")
        render_risk_summary_cards(t[COL_RISK])
        confidence_summary_cards(t[COL_CR_CONF])
        dist = risk_distribution_table(t)
        c1, c2 = st.columns([1, 1])
        with c1:
            risk_bar(t)
        with c2:
            cr_histogram(t[COL_CR])
        st.markdown("**Risk distribution**")
        st.dataframe(dist, hide_index=True, width="stretch")
        confidence_legend("combo")
        st.write("")
        show_results_table(t, height=460)
        download_xlsx("⬇️ Download results (.xlsx)",
                      {"Results": t, "Risk distribution": dist, "Model info": info_sheet},
                      "combined_batch_results.xlsx", key="dl_combo_batch", primary=True)
        st.subheader("Explanation of the risk prediction (SHAP, all rows)")
        show_shap_batch(clf_res, b["Xt"], clf_model)
