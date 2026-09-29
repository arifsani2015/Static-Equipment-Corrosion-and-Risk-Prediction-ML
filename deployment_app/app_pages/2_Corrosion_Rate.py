import pandas as pd
import streamlit as st

from info_utils import show_limitations
from model_utils import (COL_CR, COL_CR_CONF, active_reg, cr_reliability, extract_schema, predict_cr,
                         reliability_basis, similar_equipment)
from shap_utils import show_shap_batch, show_shap_single
from ui_utils import (add_history, arrange_results, build_specs, confidence_legend,
                      confidence_summary_cards, cr_block, cr_histogram, download_xlsx, file_key,
                      history_section, input_form, mode_selector, model_info_items, model_info_sheet,
                      read_batch, row_ref_note, row_from_form, show_cr_result, show_results_table, template_button,
                      to_display_columns)

ss = st.session_state
st.title("📈 Corrosion Rate Prediction (Experiment 1)")
res, model = active_reg(st)
if res is None:
    st.warning("No corrosion-rate model loaded. Open **⚙️ Settings** in the menu first.")
    st.stop()

overrides = ss["display_overrides"]
specs = build_specs([res])
num_all, _ = extract_schema(res)
st.caption(f"Active model: **{model}** | Scenario: {ss['reg_scenario']} | "
           f"Hyperparameters: {res.get('overall_source', '?')} — change in ⚙️ Settings")
show_limitations(res, model, num_all, "regression", overrides)
info_sheet = model_info_sheet(model_info_items("CR", res, model, ss.get("reg_label"), ss["reg_scenario"]))
sig = (ss.get("reg_label"), ss["reg_scenario"], model)
model_col = pd.DataFrame({"CR Model": [model]})


def predict(df, file_order=None):
    out, Xt = predict_cr(res, df, model)
    rel = cr_reliability(res, Xt, out["Predicted_CR"].values, model)
    table = arrange_results(to_display_columns(df, specs), specs, [cr_block(out["Predicted_CR"].values, rel)],
                            file_order=file_order, sort="cr" if file_order is not None else None,
                            extra_tail=pd.concat([model_col] * len(df), ignore_index=True))
    return out, Xt, rel, table


if mode_selector("cr_mode").startswith("Manual"):
    st.subheader("Equipment data")
    vals = input_form(specs, "cr")
    if st.button("🔮 Predict corrosion rate", type="primary", key="btn_cr"):
        df_in = row_from_form(specs, vals)
        out, Xt, rel, table = predict(df_in)
        ss["last_cr"] = dict(sig=sig, pred=float(out["Predicted_CR"].iloc[0]), rel=rel.iloc[0].to_dict(),
                             sim=similar_equipment(res, Xt, model, overrides), Xt=Xt, result=table)
        add_history("hist_cr", table)

    last = ss.get("last_cr")
    if last:
        if last["sig"] != sig:
            st.info("The result below was computed with a different model; press Predict again to update it.")
        st.divider()
        confidence_legend("cr")
        st.write("")
        show_cr_result(last["pred"], last["rel"], last["sim"], reliability_basis(res, model))
        download_xlsx("⬇️ Download this result (.xlsx)",
                      {"Result": last["result"], "Similar equipment": last["sim"], "Model info": info_sheet},
                      "corrosion_rate_prediction.xlsx", key="dl_cr_single", primary=True)
        st.subheader("Explanation (SHAP)")
        show_shap_single(res, last["Xt"], model)
    history_section("hist_cr", "corrosion_rate_history.xlsx", {"Model info": info_sheet})
else:
    st.subheader("Batch upload")
    st.markdown("Download the template → fill one row per equipment in sheet **Input** → upload it.")
    template_button(specs, "template_corrosion_rate.xlsx", "tmpl_cr")
    f = st.file_uploader("Upload the filled file (.xlsx / .csv)", type=["xlsx", "xls", "csv"], key="upload_batch_cr")
    if f is not None:
        cache_key = (file_key(f), sig, str(ss["default_overrides"]), str(ss["header_overrides"]), str(overrides))
        if ss.get("_cache_batch_cr", {}).get("key") != cache_key:
            with st.spinner("Predicting..."):
                df = read_batch(f, specs)
                out, Xt, rel, table = predict(df, file_order=df.attrs["file_order"])
                ss["_cache_batch_cr"] = dict(key=cache_key, table=table, Xt=Xt)
        b = ss["_cache_batch_cr"]
        t = b["table"]
        st.success(f"{len(t)} rows predicted — sorted from the highest predicted corrosion rate "
                   f"({row_ref_note(t)}).")
        confidence_summary_cards(t[COL_CR_CONF])
        confidence_legend("cr")
        st.write("")
        show_results_table(t, height=460)
        summary = pd.DataFrame({
            "Statistic": ["Rows", "Mean CR (mm/year)", "Median CR (mm/year)", "Max CR (mm/year)",
                          "High confidence", "Medium confidence", "Low confidence"],
            "Value": [len(t), round(float(t[COL_CR].mean()), 4), round(float(t[COL_CR].median()), 4),
                      round(float(t[COL_CR].max()), 4)] + [int((t[COL_CR_CONF] == k).sum()) for k in ("High", "Medium", "Low")]})
        download_xlsx("⬇️ Download results (.xlsx)", {"Results": t, "Summary": summary, "Model info": info_sheet},
                      "corrosion_rate_batch_results.xlsx", key="dl_cr_batch", primary=True)
        cr_histogram(t[COL_CR])
        st.subheader("Explanation (SHAP, all rows)")
        show_shap_batch(res, b["Xt"], model)
