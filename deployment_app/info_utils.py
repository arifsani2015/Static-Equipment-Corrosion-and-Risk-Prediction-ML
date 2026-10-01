"""'Model limitations & notes' expander shown on every prediction page, so
users know how far the model can be trusted instead of taking a single number."""
import pandas as pd
import streamlit as st

from model_utils import (get_numeric_ranges, metrics_row, quality_label, resolve_display,
                         train_score)

REG_METRIC_COLS = ["R2", "RMSE", "MAE", "IA", "R2_CV", "RMSE_CV"]
CLF_METRIC_COLS = ["Accuracy", "F1_macro", "F1_weighted", "Kappa", "ROC_AUC", "F1_macro_CV", "Accuracy_CV"]


def show_limitations(res, model_name, num_cols, kind, overrides=None):
    """kind: 'regression' or 'classification'."""
    with st.expander("📋 Model limitations & notes", expanded=False):
        row = metrics_row(res, model_name)
        q, note = quality_label(row, kind, train_score(res, model_name, kind))
        st.markdown(f"**Active model:** {model_name} (hyperparameters: {res.get('overall_source', 'unknown')}) "
                    f"— quality **{q}** ({note})")
        if row:
            present = {k: row[k] for k in (REG_METRIC_COLS if kind == "regression" else CLF_METRIC_COLS)
                       if k in row and pd.notna(row[k])}
            if present:
                st.markdown("**Performance (test set and cross-validation during training):**")
                st.dataframe(pd.DataFrame([{k: round(float(v), 4) for k, v in present.items()}]),
                             hide_index=True, width="stretch")
        ranges = get_numeric_ranges(res, num_cols)
        if ranges:
            st.markdown("**Numeric ranges seen during training** (inputs outside these ranges are "
                        "extrapolation — predictions are less reliable):")
            st.dataframe(pd.DataFrame([{"Feature": resolve_display(c, overrides), "Minimum": f"{lo:.4g}",
                                        "Maximum": f"{hi:.4g}"} for c, (lo, hi) in ranges.items()]),
                         hide_index=True, width="stretch")
        if res.get("syn_method"):
            st.markdown(f"**Data augmentation:** trained with synthetic rows ({res['syn_method']}, "
                        f"ratio {res.get('aug_ratio', '?')}x); synthetic data may represent rare/extreme "
                        f"cases less accurately.")
        st.markdown("---")
        if kind == "regression":
            st.markdown(
                "- Errors are **not uniform** across the corrosion-rate range — high CR predictions usually "
                "carry more uncertainty. Always read the prediction together with its 80% interval and "
                "confidence level.\n"
                "- The confidence level combines the model's test-set errors, agreement between all trained "
                "models and similarity to real equipment with measured CR.")
        else:
            st.markdown(
                "- The probability of the chosen class is **not a guarantee of accuracy** — models can be "
                "over- or under-confident. Look at all four probabilities and the margin to the runner-up.\n"
                "- If the corrosion rate used as input is itself a PREDICTION, its uncertainty carries over "
                "into the risk prediction.")
        st.markdown("- These models are **decision-support tools**, not a replacement for an engineering "
                    "assessment — especially for high-risk equipment.")
