# app.py – minimal UI: price + predicted revenue
import streamlit as st, pandas as pd, numpy as np, joblib

# ───────── Header ─────────
st.title("AI Price Suggestion Demo")
st.markdown("**Order Item:** 5 m³ Weight")
st.markdown("**Order Type:** Service Order")

# ───────── Load model / encoders / meta ─────────
@st.cache_resource
def load_assets():
    model = joblib.load("price_model.joblib")          # has .input_range_
    enc   = joblib.load("encoders.joblib")             # {'entity': [...], 'sp': [...], 'enc': OrdinalEncoder}
    meta  = pd.read_json("dropdown_meta.json")
    return model, enc, meta

model, enc, meta = load_assets()

# ───────── Dropdowns ─────────
entities = sorted(meta["Entity"].dropna().unique())
entity   = st.selectbox("Entity", entities)

sp_opts  = sorted(meta.loc[meta["Entity"] == entity, "Service Point"].dropna())
sp       = st.selectbox("Service Point", sp_opts)

result_box = st.empty()

# ───────── Suggest button ─────────
if st.button("Suggest Price"):

    try:
        ent_id = list(enc["entity"]).index(entity)
        sp_id  = list(enc["sp"]).index(sp)
    except ValueError:
        result_box.error("Selected context not in training data.")
        st.stop()

    prices = model.input_range_
    X = pd.DataFrame({
        "Unit Price":  prices,
        "Entity_enc":  np.full_like(prices, ent_id, dtype=float),
        "SP_enc":      np.full_like(prices,  sp_id, dtype=float)
    })
    amounts  = model.predict(X)
    revenue  = prices * amounts
    idx      = int(revenue.argmax())

    best_price   = float(prices[idx])
    best_revenue = float(revenue[idx])

    result_box.empty()
    with result_box:
        st.success(f"💶 **Suggested Price:** €{best_price:.2f}")
        st.info   (f"Predicted revenue at that price: €{best_revenue:.2f}")

# ───────── Footnote ─────────
st.caption("""
**How are these numbers calculated?**

* Demand is estimated with an XGBoost model trained on historical orders for each *Entity + Service Point* context.
* **Suggested Price** is the point in a fine price grid that maximises *price × predicted amount*.
* **Predicted revenue** is simply that price multiplied by the model-estimated amount.
""")
