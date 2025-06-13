import joblib, pandas as pd, numpy as np, streamlit as st

# ---------- Load model & encoders ----------
model     = joblib.load("price_model.joblib")
encoders  = joblib.load("encoders.joblib")
meta      = pd.read_json("dropdown_meta.json")  # names list

# ---------- UI ----------
st.title("AI Price Suggestion Demo")

# Fixed values
st.markdown("**Order Item:** 5 m³ Weight")
st.markdown("**Order Type:** Service Order")

entities = sorted(meta["entity"].unique())
entity   = st.selectbox("Entity", entities, index=0)

sps = sorted(meta.loc[meta.entity == entity, "service_point"])
sp  = st.selectbox("Service Point", sps, index=0)

if st.button("Suggest Price"):
    # encode
    ent_id = encoders["entity"].transform([[entity]])[0][0]
    sp_id  = encoders["sp"].transform([[sp]])[0][0]

    # brute-force price grid (use range saved in model attrs)
    prices  = model.input_range_        # e.g. np.linspace(...)
    X_test  = pd.DataFrame(
        {"Unit Price": prices,
         "Entity": np.full_like(prices, ent_id),
         "Service Point": np.full_like(prices, sp_id)}
    )
    amounts = model.predict(X_test)
    revenue = prices * amounts
    best    = revenue.argmax()

    st.success(f"💶 **Suggested Price:** €{prices[best]:.2f}")
    st.info(f"Predicted revenue at that price: €{revenue[best]:.2f}")
