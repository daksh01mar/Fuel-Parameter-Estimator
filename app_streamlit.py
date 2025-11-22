# app_streamlit.py
"""
Streamlit app: Predict missing fuel properties from a small input subset.
Inputs: D4052 (density), VISC (viscosity), TOTAL (sulfur ppm), FLASH (flash point)
Outputs: Predicted CN, BP50, FREEZE (using RandomForest and PLS as comparison)
If models exist they are loaded; otherwise models are trained from the dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split

# -----------------------
# Configuration / paths
# -----------------------
# Default dataset filename in repo (recommended)
DATA_PATH = "diesel_properties_clean.xlsx"

# A fallback URL from where the app will try to download the dataset if not present locally.
# NOTE: replace this with an internet-accessible URL when deploying OR include dataset in the repo.
# For quick local testing this points to your uploaded file path (will be transformed if needed).
DOWNLOAD_URL = "/mnt/data/diesel_properties_clean.xlsx"

RF_MODEL_PATH = "rf_model.joblib"
PLS_MODEL_PATH = "pls_model.joblib"
SCALER_PATH = "scaler.joblib"

RF_N_ESTIMATORS = 300
RANDOM_STATE = 42
PLS_COMPONENTS = 4

# -----------------------
# Helpers
# -----------------------
def download_dataset_if_missing(local_path=DATA_PATH, url=DOWNLOAD_URL):
    if os.path.exists(local_path):
        return local_path
    # If URL appears to be local path (starts with /), try to copy directly (works locally)
    if url.startswith("/") and os.path.exists(url):
        try:
            import shutil
            shutil.copy(url, local_path)
            return local_path
        except Exception as e:
            st.warning(f"Failed to copy local dataset from {url}: {e}")
    # Otherwise attempt HTTP download
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        return local_path
    except Exception as e:
        st.error(f"Dataset not found locally and download failed: {e}")
        return None

def load_dataset(path=DATA_PATH):
    path_used = download_dataset_if_missing(path, DOWNLOAD_URL)
    if path_used is None:
        st.stop()
    df = pd.read_excel(path_used)
    # coerce to numeric and drop fully empty rows
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all')
    return df

def train_models(df, input_cols, output_cols):
    X = df[input_cols].values
    y = df[output_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    ))
    rf.fit(X_train_s, y_train)

    pls = PLSRegression(n_components=PLS_COMPONENTS)
    pls.fit(X_train_s, y_train)

    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(pls, PLS_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return rf, pls, scaler, X_test_s, y_test, output_cols

def load_artifacts():
    rf = pls = scaler = None
    if os.path.exists(RF_MODEL_PATH) and os.path.exists(PLS_MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            rf = joblib.load(RF_MODEL_PATH)
            pls = joblib.load(PLS_MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            st.warning(f"Failed to load saved models: {e}")
            rf = pls = scaler = None
    return rf, pls, scaler

def rf_prediction_with_uncertainty(multi_rf, X_s):
    all_means = []
    all_stds = []
    for i, est in enumerate(multi_rf.estimators_):
        preds = np.stack([t.predict(X_s) for t in est.estimators_], axis=1)
        mean_pred = preds.mean(axis=1)
        std_pred = preds.std(axis=1)
        all_means.append(mean_pred)
        all_stds.append(std_pred)
    mean_preds = np.vstack(all_means).T
    std_preds = np.vstack(all_stds).T
    return mean_preds, std_preds

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Fuel Property Estimator", layout="centered")
st.title("Fuel Property Estimator (Input subset → Predict others)")

st.markdown("Enter known properties (two or more). Leave others blank. The app will predict the missing properties.")

st.info("Models: Random Forest + uncertainty (default) and PLS as comparison. Saved models will be used if present; otherwise models will be trained from dataset.")

# Load or train models
with st.spinner("Preparing models..."):
    rf_model, pls_model, scaler = load_artifacts()
    trained_here = False
    df_all = None
    if rf_model is None or pls_model is None or scaler is None:
        st.warning("Saved models not found or failed to load. Training models from dataset (this may take a minute)...")
        df_all = load_dataset()
        # Input & output plan (we will allow user to supply any two inputs; to train use the 4-input plan)
        input_cols = ["D4052", "VISC", "TOTAL", "FLASH"]
        output_cols = ["CN", "BP50", "FREEZE"]
        missing = [c for c in input_cols + output_cols if c not in df_all.columns]
        if missing:
            st.error(f"Dataset missing required columns: {missing}. Make sure the dataset contains these columns.")
            st.stop()
        rf_model, pls_model, scaler, X_test_s_validation, y_test_validation, labels = train_models(df_all, input_cols, output_cols)
        trained_here = True
    else:
        df_all = load_dataset()
        input_cols = ["D4052", "VISC", "TOTAL", "FLASH"]
        output_cols = ["CN", "BP50", "FREEZE"]
        missing = [c for c in input_cols + output_cols if c not in df_all.columns]
        if missing:
            st.error(f"Dataset missing required columns: {missing}. Can't continue.")
            st.stop()

if trained_here:
    st.success("Models trained from dataset and saved.")
else:
    st.success("Loaded saved models.")

st.sidebar.header("Enter ANY known properties (enter at least two)")
st.sidebar.write("Leave fields blank if unknown. Units must match the dataset.")

# Provide inputs (allow blank)
def side_input(name, default=""):
    val = st.sidebar.text_input(name, value=default)
    if val.strip() == "":
        return None
    try:
        return float(val)
    except:
        st.sidebar.error(f"Invalid numeric value for {name}")
        return None

# default medians from dataset to help user
medians = df_all[input_cols].median()
D4052_in = side_input("Density (D4052) [kg/m³]", str(round(float(medians["D4052"]),3)))
VISC_in = side_input("Viscosity (VISC) [mm²/s]", str(round(float(medians["VISC"]),3)))
TOTAL_in = side_input("Sulfur (TOTAL) [ppm]", str(round(float(medians["TOTAL"]),3)))
FLASH_in = side_input("Flash point (FLASH) [°C]", str(round(float(medians["FLASH"]),3)))
CN_in = side_input("Cetane Number (CN) [optional]")
BP50_in = side_input("BP50 (T50) [°C] [optional]")
FREEZE_in = side_input("Freeze point [°C] [optional]")

# Build user-provided dict
provided = {}
fields = ["CN","D4052","VISC","FLASH","BP50","FREEZE","TOTAL"]
vals = [CN_in, D4052_in, VISC_in, FLASH_in, BP50_in, FREEZE_in, TOTAL_in]
for k,v in zip(fields, vals):
    if v is not None:
        provided[k] = v

if len(provided) < 2:
    st.warning("Please provide at least two known properties to predict the rest.")
    st.stop()

# Prepare a full-row array with NaNs for missing
ALL_PROPS = ["CN","D4052","VISC","FLASH","BP50","FREEZE","TOTAL"]
x = np.full((1, len(ALL_PROPS)), np.nan)
for i,col in enumerate(ALL_PROPS):
    if col in provided:
        x[0,i] = float(provided[col])

# We will use the imputer-free approach: use existing RF/PLS models trained to predict CN,BP50,FREEZE from D4052,VISC,TOTAL,FLASH
# If user provides any subset including at least two inputs, we attempt to fill the required model inputs by:
# 1) If model input columns available (D4052,VISC,TOTAL,FLASH) present in provided, use them directly.
# 2) If some input columns are missing, we replace missing inputs by dataset medians (safe fallback) before predicting.
model_input_cols = ["D4052","VISC","TOTAL","FLASH"]
model_input = []
for col in model_input_cols:
    if col in provided:
        model_input.append(float(provided[col]))
    else:
        # median fallback
        model_input.append(float(medians[col])

        )

# Scale and predict
X_user = np.array([model_input])
X_user_s = scaler.transform(X_user)

rf_mean, rf_std = None, None
try:
    rf_mean, rf_std = rf_prediction_with_uncertainty(rf_model, X_user_s)
except Exception:
    rf_mean = rf_model.predict(X_user_s)
    rf_std = np.zeros_like(rf_mean)

rf_mean = rf_mean.flatten()
rf_std = rf_std.flatten()
pls_pred = pls_model.predict(X_user_s).flatten()

st.header("Predicted properties (from provided inputs)")
cols_display = st.columns(3)
out_labels = ["CN","BP50","FREEZE"]
for i,lab in enumerate(out_labels):
    with cols_display[i]:
        st.metric(lab, value=f"{rf_mean[i]:.3f}", delta=(f"±{rf_std[i]:.3f}" if rf_std is not None else ""))

st.markdown("**Predictions detail**")
detail = pd.DataFrame({
    "RandomForest_mean": rf_mean,
    "RandomForest_std": rf_std,
    "PLS_pred": pls_pred
}, index=out_labels)
st.table(detail)

# Export
export_df = pd.DataFrame([{**provided,
                           "CN_pred_rf": float(rf_mean[0]),
                           "BP50_pred_rf": float(rf_mean[1]),
                           "FREEZE_pred_rf": float(rf_mean[2]),
                           "CN_std_rf": float(rf_std[0]),
                           "BP50_std_rf": float(rf_std[1]),
                           "FREEZE_std_rf": float(rf_std[2]),
                           "CN_pred_pls": float(pls_pred[0]),
                           "BP50_pred_pls": float(pls_pred[1]),
                           "FREEZE_pred_pls": float(pls_pred[2])}])

if st.button("Download prediction (CSV)"):
    st.download_button("Download", export_df.to_csv(index=False).encode('utf-8'), file_name="fuel_prediction.csv", mime="text/csv")

st.success("Done. Use medians or known inputs to provide two or more properties and predict others.")
