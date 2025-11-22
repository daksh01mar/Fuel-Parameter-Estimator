# app_streamlit.py
"""
Dark-themed Streamlit app for predicting missing fuel properties.
- Enter any two or more known properties (leave the rest blank).
- The app imputes missing properties using IterativeImputer (or SimpleImputer fallback).
- OPTIONAL: run RandomForest + PLS predictor for CN, BP50, FREEZE that maps
  [D4052, VISC, TOTAL, FLASH] -> [CN, BP50, FREEZE].
- Shows predicted mean ± std (if IterativeImputer sampling enabled) or mean only.
- Uses local dataset fallback '/mnt/data/diesel_properties_clean.xlsx' (provided in your environment).
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
# Configuration & paths
# -----------------------
# Use the dataset path present in the environment (developer note): /mnt/data/diesel_properties_clean.xlsx
DOWNLOAD_URL = "/mnt/data/diesel_properties_clean.xlsx"
DATA_PATH = "diesel_properties_clean.xlsx"  # recommended: include in repo

# Saved artifacts (optional)
IMPUTER_PATH = "imputer.joblib"
IMPUTER_SCALER_PATH = "imputer_scaler.joblib"
RF_MODEL_PATH = "rf_model_subset.joblib"
PLS_MODEL_PATH = "pls_model_subset.joblib"
SCALER_PATH = "scaler_subset.joblib"

RANDOM_STATE = 42
RF_N_ESTIMATORS = 300
PLS_COMPONENTS = 4

ALL_PROPERTIES = ["CN", "D4052", "VISC", "FLASH", "BP50", "FREEZE", "TOTAL"]

# -----------------------
# Dark theme CSS
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
/* Table and text colours */
.stTable td, .stTable th {
    color: #d1d5db;
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
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# -----------------------
# Utility / data helpers
# -----------------------
def try_copy_dataset(local_path, fallback):
    """If local_path exists return it; else try to copy from fallback path (useful in hosted env)"""
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
    """Load dataset, ensure expected columns exist, coerce numeric."""
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

# -----------------------
# Imputer helpers
# -----------------------
def train_imputer(df, sample_posterior=False, n_iter=20):
    """Train IterativeImputer (or SimpleImputer fallback). Save imputer + scaler-info (mean/std)."""
    X = df.values
    # compute column-wise mean/std ignoring NaNs
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
    """Return DataFrame with n_draws imputed full-vectors (unscaled)."""
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
# Predictive models helpers (RF / PLS)
# -----------------------
def train_predict_models(df_for_models, input_cols, output_cols):
    """
    Train RandomForest + PLS mapping: input_cols -> output_cols.
    Only uses rows complete for the selected columns.
    """
    X = df_for_models[input_cols].values
    y = df_for_models[output_cols].values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    Xc = X[mask]
    yc = y[mask]
    if len(Xc) < 8:
        st.error("Not enough complete rows to train predictive models (need at least ~8).")
        st.stop()
    X_train, X_test, y_train, y_test = train_test_split(Xc, yc, test_size=0.2, random_state=RANDOM_STATE)
    scaler_model = StandardScaler()
    X_train_s = scaler_model.fit_transform(X_train)
    X_test_s = scaler_model.transform(X_test)
    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1))
    rf.fit(X_train_s, y_train)
    pls = PLSRegression(n_components=min(PLS_COMPONENTS, X_train_s.shape[1]))
    pls.fit(X_train_s, y_train)
    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(pls, PLS_MODEL_PATH)
    joblib.dump(scaler_model, SCALER_PATH)
    metrics = {
        "output_cols": output_cols,
        "rf_r2": r2_score(y_test, rf.predict(X_test_s), multioutput='raw_values').tolist(),
        "pls_r2": r2_score(y_test, pls.predict(X_test_s), multioutput='raw_values').tolist()
    }
    return rf, pls, scaler_model, metrics

def rf_predict_with_uncertainty(multi_rf, X_s):
    """Return mean and std per target by aggregating tree predictions per target."""
    per_target_means = []
    per_target_stds = []
    for est in multi_rf.estimators_:
        preds = np.stack([tree.predict(X_s) for tree in est.estimators_], axis=1)
        per_target_means.append(preds.mean(axis=1))
        per_target_stds.append(preds.std(axis=1))
    mean = np.vstack(per_target_means).T
    std = np.vstack(per_target_stds).T
    return mean, std

# -----------------------
# Page UI
# -----------------------
st.title("Fuel Parameter Predictor")
st.write("Enter known fuel parameters (at least two). Predicted missing parameters are shown below.")

# Load dataset early so we can show valid ranges under inputs
df = load_dataset()
# compute ranges and medians for hints
ranges = df[ALL_PROPERTIES].agg(["min", "max"]).to_dict()
medians = df[ALL_PROPERTIES].median().to_dict()

# Sidebar inputs with ranges shown below each input
with st.sidebar:
    st.header("Inputs (optional)")
    st.write("Provide any two or more known parameters. Leave unknown fields blank.")
    def sidebar_val_with_hint(name, col_key):
        placeholder = f"e.g. {medians[col_key]:.3f}"
        v = st.text_input(name, value="")
        hint = f"Valid range: [{ranges[col_key]['min']:.3f}  —  {ranges[col_key]['max']:.3f}]"
        st.caption(hint)
        if v is None or v.strip() == "":
            return None
        try:
            return float(v)
        except:
            st.error(f"Invalid numeric value for {name}")
            return None

    CN_in = sidebar_val_with_hint("CN (Cetane Number)", "CN")
    D4052_in = sidebar_val_with_hint("D4052 (Density) [kg/m³]", "D4052")
    VISC_in = sidebar_val_with_hint("VISC (Kinematic viscosity) [mm²/s]", "VISC")
    FLASH_in = sidebar_val_with_hint("FLASH (Flash point) [°C]", "FLASH")
    BP50_in = sidebar_val_with_hint("BP50 (T50) [°C]", "BP50")
    FREEZE_in = sidebar_val_with_hint("FREEZE (Freeze point) [°C]", "FREEZE")
    TOTAL_in = sidebar_val_with_hint("TOTAL (Sulfur) [ppm]", "TOTAL")

    st.markdown("---")
    st.write("Options")
    use_draws = st.checkbox("Estimate uncertainty using multiple imputer draws", value=True)
    n_draws = st.slider("Imputer draws", min_value=1, max_value=200, value=50) if use_draws else 1

    st.markdown("---")
    st.write("Optional predictive model (separate from imputer)")
    use_predictor = st.checkbox("Enable RandomForest/PLS predictor for CN, BP50, FREEZE", value=False)
    predictor_note = "Predictor requires at least inputs: D4052, VISC, TOTAL, FLASH (missing ones will be filled by imputer mean)."
    st.caption(predictor_note)

    run_button = st.button("Predict")

# Build provided dict
provided = {}
if CN_in is not None: provided["CN"] = CN_in
if D4052_in is not None: provided["D4052"] = D4052_in
if VISC_in is not None: provided["VISC"] = VISC_in
if FLASH_in is not None: provided["FLASH"] = FLASH_in
if BP50_in is not None: provided["BP50"] = BP50_in
if FREEZE_in is not None: provided["FREEZE"] = FREEZE_in
if TOTAL_in is not None: provided["TOTAL"] = TOTAL_in

# Run prediction
if run_button:
    if len(provided) < 2:
        st.warning("Please provide at least two known properties.")
        st.stop()

    # Ensure imputer available
    imp, scaler_info = load_imputer()
    if imp is None or scaler_info is None:
        with st.spinner("Training imputer on dataset (this may take ~30s)..."):
            imp, scaler_info = train_imputer(df, sample_posterior=(IterativeImputer is not None), n_iter=20)
        st.success("Imputer trained and saved.")

    with st.spinner("Estimating missing properties..."):
        draws_df = imputer_transform_single(imp, scaler_info, provided, n_draws=n_draws)
    mean_imputed = draws_df.mean(axis=0)
    std_imputed = draws_df.std(axis=0)

    # Display imputer results
    results = []
    for p in ALL_PROPERTIES:
        m = mean_imputed[p]
        s = std_imputed[p]
        provided_text = f"{provided[p]:.3f}" if p in provided else "-"
        std_text = f"{s:.3f}" if n_draws > 1 else "-"
        results.append({
            "Property": p,
            "Provided": provided_text,
            "Predicted Value": f"{m:.3f}",
            "Predicted Deviation": std_text
        })
    res_df = pd.DataFrame(results).set_index("Property")
    st.subheader("Imputer-based predictions")
    st.table(res_df)

    # OPTIONAL: run RF/PLS predictor for CN,BP50,FREEZE
    if use_predictor:
        st.markdown("### Predictor (RandomForest & PLS) — mapping [D4052,VISC,TOTAL,FLASH] -> [CN,BP50,FREEZE]")
        # Build model inputs: prefer provided values; else fall back to imputer mean for those inputs
        model_input_cols = ["D4052", "VISC", "TOTAL", "FLASH"]
        model_input = []
        for c in model_input_cols:
            if c in provided:
                model_input.append(float(provided[c]))
            else:
                model_input.append(float(mean_imputed[c]))

        # Load or train predictor models
        rf = pls = scaler_model = None
        if os.path.exists(RF_MODEL_PATH) and os.path.exists(PLS_MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                rf = joblib.load(RF_MODEL_PATH)
                pls = joblib.load(PLS_MODEL_PATH)
                scaler_model = joblib.load(SCALER_PATH)
            except Exception:
                rf = pls = scaler_model = None

        if rf is None or pls is None or scaler_model is None:
            with st.spinner("Training predictor models from dataset (this may take ~1min)..."):
                rf, pls, scaler_model, metrics = train_predict_models(df, model_input_cols, ["CN","BP50","FREEZE"])
            st.success("Predictor models trained and saved.")
        # Scale input and predict
        X_user = np.array([model_input])
        X_user_s = scaler_model.transform(X_user)
        rf_mean, rf_std = rf_predict_with_uncertainty(rf, X_user_s)
        rf_mean = rf_mean.flatten()
        rf_std = rf_std.flatten()
        pls_pred = pls.predict(X_user_s).flatten()

        # Present predictor outputs
        out_rows = []
        out_labels = ["CN", "BP50", "FREEZE"]
        for i, lbl in enumerate(out_labels):
            out_rows.append({
                "Target": lbl,
                "RF_mean": f"{rf_mean[i]:.3f}",
                "RF_std": f"{rf_std[i]:.3f}",
                "PLS_pred": f"{pls_pred[i]:.3f}"
            })
        out_df = pd.DataFrame(out_rows).set_index("Target")
        st.table(out_df)

        # If training-time metrics available, show R2
        if 'metrics' in locals():
            m = metrics
            metrics_df = pd.DataFrame({
                "Output": m["output_cols"],
                "RF_R2": [round(v,3) for v in m["rf_r2"]],
                "PLS_R2": [round(v,3) for v in m["pls_r2"]]
            }).set_index("Output")
            st.markdown("Predictor quick metrics (held-out R²):")
            st.table(metrics_df)

    # Download results
    payload = {}
    payload.update({f"provided_{k}": v for k, v in provided.items()})
    for p in ALL_PROPERTIES:
        payload[f"{p}_pred_mean"] = float(mean_imputed[p])
        payload[f"{p}_pred_std"] = float(std_imputed[p])
    if use_predictor:
        for i,k in enumerate(["CN","BP50","FREEZE"]):
            payload[f"{k}_rf"] = float(rf_mean[i])
            payload[f"{k}_rf_std"] = float(rf_std[i])
            payload[f"{k}_pls"] = float(pls_pred[i])
    st.download_button("Download predictions (CSV)", pd.DataFrame([payload]).to_csv(index=False).encode('utf-8'),
                       file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("This app provides screening-level predictions. For regulatory or safety-critical decisions, confirm values with laboratory measurements.")

