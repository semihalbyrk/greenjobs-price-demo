# app.py – Streamlit MVP (always show price + explicit confidence)
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
    model = joblib.load("price_model.joblib")        # includes .input_range_ & .rmse_
    enc   = joblib.load("encoders.joblib")           # {'entity': [...], 'sp': [...], 'enc': OrdinalEncoder}
    meta  = pd.read_json("dropdown_meta.json")
    return model, enc, meta

model, enc, meta = load_assets()

# ────────────── DROPDOWN UI ──────────────
entities = sorted(meta["Entity"].dropna().unique())
entity   = st.selectbox("Entity", entities)

sp_opts  = sorted(meta.loc[meta["Entity"] == entity, "Service Point"].dropna())
sp       = st.selectbox("Service Point", sp_opts)

result_box = st.empty()      # single output container

# ────────────── PREDICT BUTTON ──────────────
if st.button("Suggest Price"):

    # Map names → numeric ids (same mapping used during training)
    try:
        ent_id = list(enc["entity"]).index(entity)
        sp_id  = list(enc["sp"]).index(sp)
    except ValueError:
        result_box.error("Selected Entity / Service Point not present in training data.")
        st.stop()

    # Build candidate grid and predict
    prices = model.input_range_
    X_demo = pd.DataFrame({
        "Unit Price":  prices,
        "Entity_enc":  np.full_like(prices, ent_id, dtype=float),
        "SP_enc":      np.full_like(prices,  sp_id, dtype=float)
    })

    amounts  = model.predict(X_demo)
    revenue  = prices * amounts
    best_idx = np.argmax(revenue)

    best_price   = float(prices[best_idx])
    best_amount  = float(amounts[best_idx])
    best_revenue = float(revenue[best_idx])

    # Confidence & 95 % revenue range using model RMSE
    rmse = getattr(model, "rmse_", None)
    confidence = low_rev = high_rev = None
    if rmse is not None and best_amount > 0:
        confidence = max(0.0, 1 - rmse / best_amount)            # 0–1 scale
        lo_amt     = max(best_amount - 1.96 * rmse, 0)
        hi_amt     = best_amount + 1.96 * rmse
        low_rev    = best_price * lo_amt
        high_rev   = best_price * hi_amt

    # ────────────── OUTPUT (single block) ──────────────
    result_box.empty()
    with result_box:
        st.success(f"💶 **Suggested Price:** €{best_price:.2f}")
        st.info   (f"Predicted revenue at that price: €{best_revenue:.2f}")
        if confidence is not None:
            st.write(f"**Confidence:** {confidence:.0%}")
            st.write(f"Revenue range (95 %): €{low_rev:.2f} – €{high_rev:.2f}")
        else:
            st.write("*Confidence unavailable (RMSE not stored).*")

# ────────────── FOOTNOTE ──────────────
st.caption("""
**How are these numbers calculated?**

* Demand is estimated with an XGBoost model trained on historical orders for each *Entity + Service Point* context.
* **Suggested Price** corresponds to the maximum of *price × predicted amount* in a fine grid.
* **Confidence** is 1 – (RMSE / predicted amount). Lower model error ⇒ higher confidence.
* Revenue range shows ±1.96 × RMSE (≈ 95 % interval) converted to €.
""")
