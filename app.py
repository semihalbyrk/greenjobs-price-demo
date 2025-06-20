# app.py  – Streamlit MVP with Confidence & Revenue Range
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────── UI HEADER ───────────────────
st.title("AI Price Suggestion Demo")
st.markdown("**Order Item:** 5 m³ Weight")
st.markdown("**Order Type:** Service Order")

# ─────────────────── LOAD ASSETS (cached) ───────────────────
@st.cache_resource
def load_assets():
    model     = joblib.load("price_model.joblib")      # XGBoost with .input_range_ & .rmse_
    enc_dict  = joblib.load("encoders.joblib")         # {'entity': [...], 'sp': [...], 'enc': OrdinalEncoder}
    meta_df   = pd.read_json("dropdown_meta.json")     # list → DataFrame
    return model, enc_dict, meta_df

model, enc, meta = load_assets()

# ─────────────────── DROPDOWNS ───────────────────
entities = sorted(meta["Entity"].dropna().unique())
entity   = st.selectbox("Entity", entities)

sp_opts  = sorted(meta.loc[meta["Entity"] == entity, "Service Point"].dropna())
sp       = st.selectbox("Service Point", sp_opts)

# ─────────────────── PREDICTION BUTTON ───────────────────
if st.button("Suggest Price"):

    # map names → numeric IDs exactly as in training
    try:
        ent_id = list(enc["entity"]).index(entity)
        sp_id  = list(enc["sp"]).index(sp)
    except ValueError:
        st.error("Selected entity / service-point not found in training data.")
        st.stop()

    prices = model.input_range_                       # 100-step grid
    X_demo = pd.DataFrame({
        "Unit Price" : prices,
        "Entity_enc": np.full_like(prices, ent_id, dtype=float),
        "SP_enc"    : np.full_like(prices,  sp_id, dtype=float)
    })

    amounts  = model.predict(X_demo)
    revenue  = prices * amounts
    best_idx = revenue.argmax()

    best_price   = prices[best_idx]
    best_amount  = amounts[best_idx]
    best_revenue = revenue[best_idx]

    # ── CONFIDENCE & 95 % RANGE (simple RMSE-based) ──
    rmse = getattr(model, "rmse_", None)
    if rmse is not None:
        low_amt  = max(best_amount - 1.96 * rmse, 0)
        high_amt = best_amount + 1.96 * rmse
        low_rev  = best_price * low_amt
        high_rev = best_price * high_amt
        confidence = max(0.0, 1 - rmse / (best_amount + 1e-9))
    else:
        confidence, low_rev, high_rev = None, None, None

    # ── OUTPUT ──
    st.success(f"💶 **Suggested Price:** €{best_price:.2f}")
    st.info   (f"Predicted revenue at that price: €{best_revenue:.2f}")


# ───────── Detailed technical footnote ─────────
st.caption("""
### How are these numbers produced?

1. **Training data**  
   Historical orders for a single Order Item are filtered by Entity & Service Point.  
   Each record contributes: `Unit Price`, `Entity_ID`, `ServicePoint_ID` ➜ *Actual Amount*.

2. **Model**  
   A gradient-boosted tree regressor (XGBoost) is trained to learn  
   \\(\\text{Amount} = f(\\text{Price},\\,\\text{Entity},\\,\\text{Service Point})\\).  
   The model’s hold-out RMSE is stored inside `price_model.joblib` as `model.rmse_`.

3. **Price scan at runtime**  
   *100* candidate prices are generated between the min & max prices seen in training  
   (`model.input_range_`).  The model predicts an amount for each candidate while  
   holding the encoded `Entity` & `Service Point` constant.

4. **Suggested Price**  
   For each candidate we compute **Predicted Revenue** = *price × predicted amount*.  
   The price that maximises this revenue curve is returned as **Suggested Price**.

5. **Confidence & 95 % band (currently hidden)**  
   When enabled, confidence \\(C\\approx 1 - \\text{RMSE}/\\hat{y}\\) expresses how large the
   error is relative to the predicted amount.  
   An approximate 95 % revenue range is derived from  
   \\(\\hat{y} \\pm 1.96·\\text{RMSE}\\).

> **Limitations**  
> * Works best if the historical data for the chosen Entity + Service Point contains
>   multiple price points.  
> * Seasonality, time trends and external costs are **not** modelled in this MVP.
""")
