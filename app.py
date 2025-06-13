import json, numpy as np, pandas as pd, joblib, streamlit as st

# ---------- Load ----------
model    = joblib.load("price_model.joblib")
enc_dict = joblib.load("encoders.joblib")      # contains categories & encoder
meta     = pd.read_json("dropdown_meta.json")  # ← JSON → DataFrame

# ---- build dropdowns ----
entities = sorted(meta["Entity"].dropna().unique())
entity   = st.selectbox("Entity", entities)

sp_opts  = sorted(meta.loc[meta["Entity"] == entity, "Service Point"].dropna())
sp       = st.selectbox("Service Point", sp_opts)

if st.button("Suggest Price"):
    # map names → numeric ids (same order as training)
    ent_id = list(enc_dict["entity"]).index(entity)
    sp_id  = list(enc_dict["sp"]).index(sp)

    prices = model.input_range_
    X_demo = pd.DataFrame({
        "Unit Price": prices,
        "Entity_enc": np.full_like(prices, ent_id, dtype=float),
        "SP_enc":     np.full_like(prices, sp_id,  dtype=float)
    })
    amounts  = model.predict(X_demo)
    revenue  = prices * amounts
    best_idx = revenue.argmax()

    st.success(f"💶 **Suggested Price:** €{prices[best_idx]:.2f}")
    st.info   (f"Predicted revenue at that price: €{revenue[best_idx]:.2f}")
