"""
Corrosion Rate & Risk Prediction -- demo app
=============================================
Run locally:   streamlit run app.py
Run on Kaggle: see ../Jalankan_di_Kaggle.ipynb
Run online:    push this folder to GitHub and deploy on https://streamlit.io/cloud

No login needed. Model files (checkpoints), model selection and the input
schema are managed on the "Settings" page.
"""
import streamlit as st

from model_utils import active_clf, active_reg, auto_load_once, init_session_state

st.set_page_config(page_title="Corrosion Rate & Risk Prediction", page_icon="🛢️", layout="wide")
init_session_state(st)

# Auto-detect Experiment 1 & 2 checkpoints (Kaggle notebook Outputs / Datasets
# attached to the session, or the local ./checkpoints folder) -- ONCE per
# session, on whichever page is opened first.
if not st.session_state["_auto_discover_tried"]:
    with st.spinner("Looking for Experiment 1 & 2 checkpoints..."):
        auto_load_once(st)

settings = st.Page("app_pages/1_Settings.py", title="Settings", icon="⚙️")
pred_cr = st.Page("app_pages/2_Corrosion_Rate.py", title="Corrosion Rate Prediction", icon="📈")
pred_risk = st.Page("app_pages/3_Risk.py", title="Risk Prediction", icon="⚠️")
combined = st.Page("app_pages/4_Combined.py", title="Combined (CR → Risk)", icon="🔗")

nav = st.navigation({"Setup": [settings], "Prediction": [pred_cr, pred_risk, combined]})

st.sidebar.divider()
st.sidebar.caption("Active models")
reg_res, reg_model = active_reg(st)
clf_res, clf_model = active_clf(st)
st.sidebar.write(("✅" if reg_res else "❌") + " Corrosion rate (Experiment 1)")
if reg_res:
    st.sidebar.caption(f"{reg_model} · scenario {st.session_state['reg_scenario']}  \n"
                       f"{st.session_state.get('reg_label') or 'manual upload'}")
st.sidebar.write(("✅" if clf_res else "❌") + " Risk (Experiment 2)")
if clf_res:
    st.sidebar.caption(f"{clf_model}"
                       + (f" · CR {st.session_state['clf_source']}" if st.session_state.get("clf_source") else "")
                       + f"  \n{st.session_state.get('clf_label') or 'manual upload'}")
nav.run()
