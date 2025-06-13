# app.py – Streamlit MVP (single-output, confidence gating)
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ────────────── HEADER ──────────────
st.title("AI Price Suggestion Demo")
st.markdown("**Order Item:** 5 m³ Weight")
st.markdown("**Order Type:** Service Order")

# ────────────── ASSET LOADER ──────────────
@st.cache_resource
def load_assets():
    model = joblib.load("price_model.joblib")           # has .input_range_ & .rmse_
    enc   = joblib.load("encoders.joblib")              # {'entity': [...], 'sp': [...], 'enc': OrdinalEncoder}
    meta  = pd.read_json("dropdown_meta.json")
    return model, enc, meta

model, enc, meta = load_assets()

# ────────────── DROPDOWNS ──────────────
entities = sorted(meta["Entity"].dropna().unique())
entity   = st.selectbox("Entity", entities)

sp_opts  = sorted(meta.loc[meta["Entity"] == entity, "Service Point"].dropna())
sp       = st.selectbox("Service Point", sp_opts)

# ────────────── RESULT PLACEHOLDER ──────────────
result_box = st.empty()

# ────────────── BUTTON ──────────────
if st.button("Suggest Price"):

    # map names ➜ encoded ids
    try:
        ent_id = list(enc["entity"]).index(entity)
        sp_id  = list(enc["sp"]).index(sp)
    except ValueError:
        result_box.error("Selected Entity / Service Point not present in training data.")
        st.stop()

    # candidate grid
    prices = model.input_range_
    X_demo = pd.DataFrame({
        "Unit Price":  prices,
        "Entity_enc":  np.full_like(prices, ent_id, dtype=float),
        "SP_enc":      np.full_like(prices,  sp_id, dtype=float)
    })

    amounts  = model.predict(X_demo)
    revenue  = prices * amounts
    best_idx = revenue.argmax()

    best_price   = float(prices[best_idx])
    best_amount  = float(amounts[best_idx])
    best_revenue = float(revenue[best_idx])

    # confidence & range
    rmse = getattr(model, "rmse_", None)
    confidence = low_rev = high_rev = None
    if rmse is not None and best_amount > 0:
        confidence = max(0.0, 1 - rmse / best_amount)
        lo_amt     = max(best_amount - 1.96 * rmse, 0)
        hi_amt     = best_amount + 1.96 * rmse
        low_rev    = best_price * lo_amt
        high_rev   = best_price * hi_amt

    # threshold guard
    CONF_THRESHOLD = 0.50
    result_box.empty()
    if confidence is not None and confidence < CONF_THRESHOLD:
        result_box.warning(
            f"Insufficient data for a reliable estimate in this context "
            f"(confidence {confidence:.0%}).\n\n"
            "Please gather more orders for this Entity + Service Point "
            "before using the price optimiser."
        )
        st.stop()

    # output (single set)
    with result_box:
        st.success(f"💶 **Suggested Price:** €{best_price:.2f}")
        st.info   (f"Predicted revenue at that price: €{best_revenue:.2f}")
        if confidence is not None:
            st.write(f"**Confidence:** {confidence:.0%}")
            st.write(f"Revenue range (95 %): €{low_rev:.2f} – €{high_rev:.2f}")

# ────────────── FOOTNOTE ──────────────
st.caption("""
**How are these numbers calculated?**

* The model (XGBoost) is trained on historical orders to estimate demand (amount)
  as a function of price for each *Entity + Service Point* context.
* **Suggested Price** is the point in a fine price grid that maximises
  *price × predicted amount*.
* **Confidence** is derived from the model’s RMSE:
  a lower error means a higher confidence score.
* The 95 % revenue range applies ±1.96 × RMSE to the predicted amount and converts it to €.
""")
