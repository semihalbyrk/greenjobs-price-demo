# app.py  – Streamlit MVP (price + revenue, confidence in %)
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ────────────── HEADER ──────────────
st.title("AI Price Suggestion Demo")
st.markdown("**Order Item:** 5 m³ Weight")
st.markdown("**Order Type:** Service Order")

# ────────────── LOAD ARTEFACTS ──────────────
@st.cache_resource
def load_assets():
    model = joblib.load("price_model.joblib")          # has .input_range_ & .rmse_
    enc   = joblib.load("encoders.joblib")             # {'entity': [...], 'sp': [...], 'enc': OrdinalEncoder}
    meta  = pd.read_json("dropdown_meta.json")
    return model, enc, meta

model, enc, meta = load_assets()

# ────────────── DROPDOWNS ──────────────
entities = sorted(meta["Entity"].dropna().unique())
entity   = st.selectbox("Entity", entities)

sp_opts  = sorted(meta.loc[meta["Entity"] == entity, "Service Point"].dropna())
sp       = st.selectbox("Service Point", sp_opts)

# single output container
result_box = st.empty()

# ────────────── BUTTON ──────────────
if st.button("Suggest Price"):

    # map names → ids
    ent_id = list(enc["entity"]).index(entity) if entity in enc["entity"] else None
    sp_id  = list(enc["sp"]).index(sp)        if sp     in enc["sp"]     else None
    if ent_id is None or sp_id is None:
        result_box.error("Selected Entity / Service Point not present in training data.")
        st.stop()

    prices = model.input_range_
    X_demo = pd.DataFrame({
        "Unit Price":  prices,
        "Entity_enc":  np.full_like(prices, ent_id, dtype=float),
        "SP_enc":      np.full_like(prices,  sp_id, dtype=float)
    })

    amounts  = model.predict(X_demo)
    revenue  = prices * amounts
    best_idx = int(np.argmax(revenue))

    best_price   = float(prices[best_idx])
    best_amount  = float(amounts[best_idx])
    best_revenue = float(revenue[best_idx])

    # confidence
    rmse = getattr(model, "rmse_", None)
    confidence = None
    if rmse is not None and best_amount > 0:
        confidence = max(0.0, 1 - rmse / best_amount)     # 0–1 scale → %
        conf_txt   = f" ({confidence:.0%} conf.)"
    else:
        conf_txt   = ""

    # output
    result_box.empty()
    with result_box:
        st.success(f"💶 **Suggested Price:** €{best_price:.2f}{conf_txt}")
        st.info   (f"Predicted revenue at that price: €{best_revenue:.2f}")

# ────────────── FOOTNOTE ──────────────
st.caption("""
*Suggested Price* maximises *price × predicted demand* according to an XGBoost model.
The confidence percentage is derived from the model’s RMSE:  
lower error ⇒ higher confidence.
""")
