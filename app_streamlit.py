# app_streamlit.py
"""
Dark-themed Streamlit app for predicting missing fuel properties.
- Enter any two or more known properties (leave the rest blank).
- The app imputes missing properties using IterativeImputer (or SimpleImputer fallback).
- Shows predicted mean ± std (if IterativeImputer with sampling enabled) or mean only.
- Uses local dataset 'diesel_properties_clean.xlsx' by default (included in repo or copied from fallback).
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Robust import for IterativeImputer (works across sklearn versions)
try:
    from sklearn.impute import IterativeImputer
except Exception:
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
    except Exception:
        IterativeImputer = None

from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -----------------------
# Configuration
# -----------------------
# dataset fallback local path (use the uploaded file path)
DOWNLOAD_URL = "/mnt/data/diesel_properties_clean.xlsx"
DATA_PATH = "diesel_properties_clean.xlsx"  # recommended to include in repo

IMPUTER_PATH = "imputer.joblib"
IMPUTER_SCALER_PATH = "imputer_scaler.joblib"
RF_MODEL_PATH = "rf_model_subset.joblib"
PLS_MODEL_PATH = "pls_model_subset.joblib"
SCALER_PATH = "scaler_subset.joblib"

RANDOM_STATE = 42

ALL_PROPERTIES = ["CN", "D4052", "VISC", "FLASH", "BP50", "FREEZE", "TOTAL"]

# -----------------------
# UI theme & CSS (dark)
# -----------------------
st.set_page_config(page_title="Fuel Parameter Predictor", layout="centered", initial_sidebar_state="expanded")

dark_css = """
<style>
/* Background and fonts */
body, .stApp, .main {
    background-color: #0f1720;
    color: #d1d5db;
}
/* Sidebar */
.reportview-container .sidebar-content {
    background-color: #0b1220;
    color: #d1d5db;
}
/* Headers */
h1, h2, h3, h4, h5 {
    color: #e6eef8;
}
/* Metric boxes and table styling */
.stTable td, .stTable th {
    color: #d1d5db;
}
.metric-box {
    background-color: #0b1220;
    border: 1px solid #1f2937;
    padding: 12px;
    border-radius: 8px;
}
/* Buttons */
.stButton>button {
    background-color: #1f2937;
    color: #d1d5db;
    border-radius: 8px;
    padding: 0.5rem 1rem;
}
.stButton>button:hover {
    background-color: #111827;
}
/* Inputs */
div[data-baseweb="input"] input {
    background-color: #0b1220;
    color: #d1d5db;
    border: 1px solid #374151;
    border-radius: 6px;
}
/* Download button */
.css-1lsmgbg.egzxvld0 { background-color:#111827; }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# -----------------------
# Helper functions
# -----------------------
def try_copy_dataset(local_path, fallback):
    if os.path.exists(local_path):
        return local_path
    if fallback and os.path.exists(fallback):
        import shutil
        try:
            shutil.copy(fallback, local_path)
            return local_path
        except Exception:
            return None
    return None

def load_dataset(path=DATA_PATH, fallback=DOWNLOAD_URL):
    p = try_copy_dataset(path, fallback)
    if not p:
        if os.path.exists(fallback):
            p = fallback
        else:
            st.error("Dataset not found. Put 'diesel_properties_clean.xlsx' in the app folder or update DATA_PATH.")
            st.stop()
    df = pd.read_excel(p)
    missing = [c for c in ALL_PROPERTIES if c not in df.columns]
    if missing:
        st.error(f"Dataset is missing expected columns: {missing}")
        st.stop()
    df = df[ALL_PROPERTIES].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    return df

def train_imputer(df, sample_posterior=False, n_iter=20):
    X = df.values
    col_mean = np.nanmean(X, axis=0)
    col_std = np.nanstd(X, axis=0)
    col_std[col_std == 0] = 1.0
    Xs = (X - col_mean) / col_std
    if IterativeImputer is not None:
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=n_iter,
                               sample_posterior=sample_posterior, random_state=RANDOM_STATE,
                               initial_strategy="mean")
        imp.fit(Xs)
    else:
        imp = SimpleImputer(strategy="mean")
        imp.fit(Xs)
    joblib.dump(imp, IMPUTER_PATH)
    scaler_info = {"mean": col_mean, "std": col_std}
    joblib.dump(scaler_info, IMPUTER_SCALER_PATH)
    return imp, scaler_info

def load_imputer():
    if os.path.exists(IMPUTER_PATH) and os.path.exists(IMPUTER_SCALER_PATH):
        imp = joblib.load(IMPUTER_PATH)
        scaler_info = joblib.load(IMPUTER_SCALER_PATH)
        return imp, scaler_info
    return None, None

def imputer_transform_single(imp, scaler_info, row_dict, n_draws=1):
    x = np.full((1, len(ALL_PROPERTIES)), np.nan, dtype=float)
    for i, col in enumerate(ALL_PROPERTIES):
        if col in row_dict and row_dict[col] is not None:
            x[0, i] = float(row_dict[col])
    mean = scaler_info["mean"]
    std = scaler_info["std"]
    x_s = (x - mean) / std
    draws = []
    for _ in range(n_draws):
        x_imp_s = imp.transform(x_s)
        x_imp = (x_imp_s * std) + mean
        draws.append(x_imp.flatten())
    df_draws = pd.DataFrame(draws, columns=ALL_PROPERTIES)
    return df_draws

# -----------------------
# Page content
# -----------------------
st.title("Fuel Parameter Predictor")
st.write("Enter known fuel parameters (at least two). The app predicts the remaining parameters using multivariate imputation.")

# Sidebar inputs
with st.sidebar:
    st.header("Inputs (optional)")
    st.write("Provide any two or more known parameters. Leave unknown fields blank.")
    def sidebar_val(name, placeholder=""):
        v = st.text_input(name, value=placeholder)
        if v is None or v.strip() == "":
            return None
        try:
            return float(v)
        except:
            st.error(f"Invalid numeric value for {name}")
            return None

    CN_in = sidebar_val("CN (Cetane Number)")
    D4052_in = sidebar_val("D4052 (Density) [kg/m³]")
    VISC_in = sidebar_val("VISC (Kinematic viscosity) [mm²/s]")
    FLASH_in = sidebar_val("FLASH (Flash point) [°C]")
    BP50_in = sidebar_val("BP50 (T50) [°C]")
    FREEZE_in = sidebar_val("FREEZE (Freeze point) [°C]")
    TOTAL_in = sidebar_val("TOTAL (Sulfur) [ppm]")

    st.markdown("---")
    st.write("Options")
    use_draws = st.checkbox("Estimate uncertainty using multiple imputer draws", value=True)
    n_draws = st.slider("Imputer draws", min_value=1, max_value=200, value=50) if use_draws else 1
    run_button = st.button("Predict")

# load dataset
df = load_dataset()

# Build provided dict
provided = {}
if CN_in is not None: provided["CN"] = CN_in
if D4052_in is not None: provided["D4052"] = D4052_in
if VISC_in is not None: provided["VISC"] = VISC_in
if FLASH_in is not None: provided["FLASH"] = FLASH_in
if BP50_in is not None: provided["BP50"] = BP50_in
if FREEZE_in is not None: provided["FREEZE"] = FREEZE_in
if TOTAL_in is not None: provided["TOTAL"] = TOTAL_in

if run_button:
    if len(provided) < 2:
        st.warning("Please provide at least two known properties.")
        st.stop()

    # ensure imputer exists or train it
    imp, scaler_info = load_imputer()
    if imp is None or scaler_info is None:
        with st.spinner("Training imputer on dataset..."):
            imp, scaler_info = train_imputer(df, sample_posterior=(IterativeImputer is not None), n_iter=30)
        st.success("Imputer trained and saved.")

    with st.spinner("Estimating missing properties..."):
        draws_df = imputer_transform_single(imp, scaler_info, provided, n_draws=n_draws)
    mean_imputed = draws_df.mean(axis=0)
    std_imputed = draws_df.std(axis=0)

    # Display results in a clean table
    results = []
    for p in ALL_PROPERTIES:
        m = mean_imputed[p]
        s = std_imputed[p]
        if n_draws <= 1:
            std_text = "-"
        else:
            std_text = f"{s:.3f}"
        provided_text = f"{provided[p]:.3f}" if p in provided else "-"
        results.append({
            "Property": p,
            "Provided": provided_text,
            "Predicted_mean": f"{m:.3f}",
            "Predicted_std": std_text
        })
    res_df = pd.DataFrame(results).set_index("Property")

    st.subheader("Predictions (imputer)")
    st.table(res_df)

    # Optionally show scatter plots for context (density vs others)
    if st.checkbox("Show exploratory plots (training data)"):
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        axs[0].scatter(df["D4052"], df["CN"], alpha=0.6, color="#9CA3AF")
        axs[0].set_xlabel("D4052"); axs[0].set_ylabel("CN")
        axs[1].scatter(df["D4052"], df["BP50"], alpha=0.6, color="#9CA3AF")
        axs[1].set_xlabel("D4052"); axs[1].set_ylabel("BP50")
        axs[2].scatter(df["D4052"], df["FREEZE"], alpha=0.6, color="#9CA3AF")
        axs[2].set_xlabel("D4052"); axs[2].set_ylabel("FREEZE")
        fig.patch.set_facecolor('#0f1720')
        for ax in axs:
            ax.set_facecolor('#0b1220')
            ax.tick_params(colors='#d1d5db')
            ax.xaxis.label.set_color('#d1d5db')
            ax.yaxis.label.set_color('#d1d5db')
        st.pyplot(fig)

    # Download button
    out_payload = {}
    out_payload.update({f"provided_{k}": v for k, v in provided.items()})
    for p in ALL_PROPERTIES:
        out_payload[f"{p}_pred_mean"] = float(mean_imputed[p])
        out_payload[f"{p}_pred_std"] = float(std_imputed[p])
    pd_out = pd.DataFrame([out_payload])
    st.download_button("Download predictions (CSV)", pd_out.to_csv(index=False).encode('utf-8'), file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("This app provides screening-level predictions. For critical decisions, confirm with laboratory tests.")
