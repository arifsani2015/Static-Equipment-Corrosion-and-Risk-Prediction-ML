"""SHAP displays shared by the Corrosion Rate, Risk and Combined pages
(responsive Altair charts)."""
import streamlit as st

from model_utils import compute_shap
from ui_utils import shap_chart


def show_shap_single(res, Xt_row, model_name=None):
    df_sv = compute_shap(res, Xt_row, model_name, how="signed")
    if df_sv is None:
        st.info("SHAP is not available for this model.")
        return
    row = df_sv.iloc[0]
    row = row.reindex(row.abs().sort_values(ascending=False).index).head(10)
    title = ("Why the model predicted this risk class (top 10 features)" if "present" in res
             else "Why the model predicted this corrosion rate (top 10 features)")
    shap_chart(row, title, signed=True)


def show_shap_batch(res, Xt, model_name=None, max_rows=500):
    df_sv = compute_shap(res, Xt.head(max_rows), model_name, how="abs")
    if df_sv is None:
        st.info("SHAP is not available for this model.")
        return
    imp = df_sv.mean().sort_values(ascending=False).head(12)
    shap_chart(imp, "Average feature importance" + (f" (first {max_rows} rows)" if len(Xt) > max_rows else ""),
               signed=False)
