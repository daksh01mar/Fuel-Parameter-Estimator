# app_streamlit.py
"""
Styled Streamlit App for Fuel Property Estimation
Features:
 - Accepts any subset of fuel properties (enter at least 2)
 - Uses IterativeImputer to fill missing properties
 - Trains/loads RandomForest and PLS models to predict CN/BP50/FREEZE
 - Computes PASS/CHECK/FAIL based on spec limits with safety margins
 - Quality Index (0-100)
 - Saves and allows CSV download
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ---------------------------
# Configuration - edit if needed
# ---------------------------
DATA_PATH = "diesel_properties_clean.xlsx"   # recommended: include this file in repo
DOWNLOAD_URL = "/mnt/data/diesel_properties_clean.xlsx"  # local fallback path (used when running locally)
IMPUTER_PATH = "imputer.joblib"
IMPUTER_SCALER_PATH = "imputer_scaler.joblib"
RF_MODEL_PATH = "rf_model_subset.joblib"
PLS_MODEL_PATH = "pls_model_subset.joblib"
SCALER_PATH = "scaler_subset.joblib"

RANDOM_STATE = 42
RF_N_ESTIMATORS = 300
PLS_COMPONENTS = 4

# These are the full list of properties expected in the dataset
ALL_PROPERTIES = ["CN", "D4052", "VISC", "FLASH", "BP50", "FREEZE", "TOTAL"]

# Spec limits (example: EN590 / IS 1460 typical). Edit as required for your standard.
SPEC_LIMITS = {
    "CN": {"min": 51, "direction": "gte"},
    "D4052": {"min": 820, "max": 845, "direction": "in"},
    "VISC": {"min": 2.0, "max": 4.5, "direction": "in"},
    "FLASH": {"min": 55, "direction": "gte"},
    "BP50": {"min": 245, "max": 350, "direction": "in"},
    "FREEZE": {"max": -20, "direction": "lte"},
    "TOTAL": {"max": 10, "direction": "lte"}
}

# Weighting for quality index (must sum to 1)
QUALITY_WEIGHTS = {
    "CN": 0.20,
    "TOTAL": 0.20,
    "VISC": 0.15,
    "D4052": 0.15,
    "FLASH": 0.10,
    "BP50": 0.10,
    "FREEZE": 0.10
}

# Safety margins: used to make automated PASS conservative (reduce false positives)
SAFETY_MARGINS = {
    "CN": 2.0,      # require predictions to be >= (spec + margin) for auto PASS
    "TOTAL": 0.0,
    "D4052": 0.0,
    "VISC": 0.0,
    "FLASH": 2.0,
    "BP50": 0.0,
    "FREEZE": 0.0
}

# ---------------------------
# Helper functions
# ---------------------------
def local_css():
    css = """
    <style>
    .stApp { background-color: #F8F9FB; color: #17202A; }
    .header {text-align: center;}
    .metric-box { background-color:#ffffff; border-radius:10px; padding:10px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
    .small-muted { color:#566573; font-size:12px; }
    .big-number { font-size:22px; font-weight:600; color:#1B4F72; }
    .card { background: linear-gradient(180deg,#ffffff,#fbfdff); padding:10px; border-radius:10px; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

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
    # Try local path, otherwise fallback copy (works in some local setups)
    p = try_copy_dataset(path, fallback)
    if not p:
        # try reading fallback directly (if fallback is a URL this would have to be downloaded)
        if os.path.exists(fallback):
            p = fallback
        else:
            st.error("Dataset not found. Put 'diesel_properties_clean.xlsx' in the app folder or adjust DATA_PATH.")
            st.stop()
    df = pd.read_excel(p)
    # Keep only expected columns (coerce to numeric)
    missing = [c for c in ALL_PROPERTIES if c not in df.columns]
    if missing:
        st.error(f"Dataset does not contain expected columns: {missing}")
        st.stop()
    df = df[ALL_PROPERTIES].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")  # drop completely empty rows
    return df

def train_imputer(df, sample_posterior=False, n_iter=20):
    # Standardize before training imputer
    scaler = StandardScaler()
    X = df.values
    # Fit scaler ignoring NaNs: StandardScaler cannot handle NaNs; perform column-wise mean sub and division by std
    # We'll compute mean/std ignoring NaN
    col_mean = np.nanmean(X, axis=0)
    col_std = np.nanstd(X, axis=0)
    # Avoid zero-std
    col_std[col_std == 0] = 1.0
    Xs = (X - col_mean) / col_std
    imp = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=n_iter,
        sample_posterior=sample_posterior,
        random_state=RANDOM_STATE,
        initial_strategy="mean"
    )
    imp.fit(Xs)
    # Save scaler params as dict
    scaler_info = {"mean": col_mean, "std": col_std}
    joblib.dump(imp, IMPUTER_PATH)
    joblib.dump(scaler_info, IMPUTER_SCALER_PATH)
    return imp, scaler_info

def load_imputer():
    if os.path.exists(IMPUTER_PATH) and os.path.exists(IMPUTER_SCALER_PATH):
        imp = joblib.load(IMPUTER_PATH)
        scaler_info = joblib.load(IMPUTER_SCALER_PATH)
        return imp, scaler_info
    return None, None

def imputer_transform_single(imp, scaler_info, row_dict, n_draws=1):
    """
    row_dict: mapping property->value or missing
    n_draws: if imputer was trained with sample_posterior=True, multiple draws give uncertainty
    returns DataFrame with n_draws rows of full properties (unscaled)
    """
    # Build input row with NaNs where missing
    x = np.full((1, len(ALL_PROPERTIES)), np.nan, dtype=float)
    for i, col in enumerate(ALL_PROPERTIES):
        if col in row_dict and row_dict[col] is not None:
            x[0, i] = float(row_dict[col])
    # scale using scaler_info
    mean = scaler_info["mean"]
    std = scaler_info["std"]
    x_s = (x - mean) / std
    draws = []
    for _ in range(n_draws):
        x_imp_s = imp.transform(x_s)  # returns scaled
        x_imp = (x_imp_s * std) + mean
        draws.append(x_imp.flatten())
    df_draws = pd.DataFrame(draws, columns=ALL_PROPERTIES)
    return df_draws

def train_predict_models(df_for_models, input_cols, output_cols):
    # Train RF + PLS to map input_cols -> output_cols
    X = df_for_models[input_cols].values
    y = df_for_models[output_cols].values
    # Drop rows with NaNs in these columns (train on complete rows)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    Xc = X[mask]
    yc = y[mask]
    if len(Xc) < 8:
        st.error("Not enough complete rows to train the predictive models (need >8).")
        st.stop()
    X_train, X_test, y_train, y_test = train_test_split(Xc, yc, test_size=0.2, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1))
    rf.fit(X_train_s, y_train)
    pls = PLSRegression(n_components=min(PLS_COMPONENTS, X_train_s.shape[1]))
    pls.fit(X_train_s, y_train)
    # Save artifacts
    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(pls, PLS_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    # Evaluate quickly
    y_rf = rf.predict(X_test_s)
    y_pls = pls.predict(X_test_s)
    metrics = {
        "rf_r2": r2_score(y_test, y_rf, multioutput='raw_values').tolist(),
        "pls_r2": r2_score(y_test, y_pls, multioutput='raw_values').tolist(),
        "output_cols": output_cols
    }
    return rf, pls, scaler, metrics

def rf_predict_with_uncertainty(multi_rf, X_s):
    # For each output, collect predictions from each base estimator and compute mean/std
    per_target_means = []
    per_target_stds = []
    for est in multi_rf.estimators_:
        # each est is RandomForestRegressor
        preds = np.stack([tree.predict(X_s) for tree in est.estimators_], axis=1)  # (n_samples, n_trees)
        per_target_means.append(preds.mean(axis=1))
        per_target_stds.append(preds.std(axis=1))
    mean = np.vstack(per_target_means).T
    std = np.vstack(per_target_stds).T
    return mean, std

def compute_quality_from_vector(vec_dict, weights=QUALITY_WEIGHTS, specs=SPEC_LIMITS, margins=SAFETY_MARGINS):
    """
    vec_dict: mapping properties -> numeric values (predicted or actual)
    returns quality_score (0-100), pass_fail ('PASS'/'CHECK'/'FAIL'), details
    """
    # per-property normalized scores (0..1)
    scores = {}
    failed_any = False
    borderline = False
    for prop, w in weights.items():
        val = vec_dict.get(prop, np.nan)
        spec = specs.get(prop, {})
        direction = spec.get("direction", None)
        s = 0.0
        if np.isnan(val):
            s = 0.0
        else:
            if direction == "gte":
                minv = spec["min"]
                s = np.clip((val - minv) / ( (minv*0.2) if (minv*0.2)>0 else 10 ), 0, 1)
                # check safety margin
                if val < (minv + margins.get(prop, 0)):
                    failed_any = True if val < minv else failed_any
                    if val < (minv + margins.get(prop, 0)):
                        borderline = True if val >= minv else borderline
            elif direction == "lte":
                maxv = spec["max"]
                s = np.clip((maxv - val) / ( (abs(maxv)*0.2) if (abs(maxv)*0.2)>0 else 10 ), 0, 1)
                if val > (maxv - margins.get(prop, 0)):
                    failed_any = True if val > maxv else failed_any
                    if val <= maxv and val > (maxv - margins.get(prop, 0)):
                        borderline = True
            elif direction == "in":
                minv = spec["min"]; maxv = spec["max"]
                if minv <= val <= maxv:
                    s = 1.0
                else:
                    # linear falloff, use 10% range as tolerance
                    tol = max((maxv-minv)*0.2, 1.0)
                    if val < minv:
                        s = np.clip(1 - ((minv - val) / tol), 0, 1)
                    else:
                        s = np.clip(1 - ((val - maxv) / tol), 0, 1)
                # safety margin: require inner margin for auto PASS
                if val < (minv + margins.get(prop,0)) or val > (maxv - margins.get(prop,0)):
                    borderline = True
                    if val < minv or val > maxv:
                        failed_any = True
            else:
                s = 0.0
        scores[prop] = s
    # weighted average
    total_weight = sum([weights[k] for k in weights if not np.isnan(scores.get(k, np.nan))])
    if total_weight == 0:
        quality_index = 0.0
    else:
        quality_index = sum([scores[k]*weights[k] for k in weights]) / total_weight
    quality_score_100 = round(float(quality_index*100), 2)
    # Decide PASS / CHECK / FAIL
    # If any property truly outside spec -> FAIL
    # If not fail but any borderline -> CHECK
    # Else PASS
    real_fail = False
    for prop, spec in SPEC_LIMITS.items():
        val = vec_dict.get(prop, np.nan)
        if np.isnan(val):
            continue
        dir = spec.get("direction")
        if dir == "gte" and val < spec["min"]:
            real_fail = True
        if dir == "lte" and val > spec["max"]:
            real_fail = True
        if dir == "in" and not (spec["min"] <= val <= spec["max"]):
            real_fail = True
    if real_fail:
        label = "FAIL"
    elif borderline:
        label = "CHECK"
    else:
        label = "PASS"
    return quality_score_100, label, scores

# ---------------------------
# App UI - Build page
# ---------------------------
st.set_page_config(page_title="Fuel Quality Estimator", layout="centered", initial_sidebar_state="expanded")
local_css()
st.markdown("<div class='header'><h1 style='color:#2E86C1;'>🔥 Fuel Quality Estimator</h1></div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#566573;'>Enter any two or more known fuel parameters — the app will infer the rest and show PASS / CHECK / FAIL</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar - inputs
with st.sidebar:
    st.markdown("<div style='padding:12px;border-radius:8px;background:#EBF5FB;'>"
                "<h3 style='color:#117864;'>🔧 Known Properties (optional)</h3>"
                "<p class='small-muted'>Enter values you know. Leave unknown fields empty.</p></div>",
                unsafe_allow_html=True)
    # Provide text inputs (allow blank)
    def input_val(name, default=""):
        v = st.text_input(name, value=default)
        if v is None or v.strip() == "":
            return None
        try:
            return float(v)
        except:
            st.error(f"Invalid numeric value for {name}")
            return None

    # Will set defaults to medians once df loaded; for now empty
    CN_in = input_val("Cetane Number (CN)")
    D4052_in = input_val("Density (D4052) [kg/m³]")
    VISC_in = input_val("Kinematic Viscosity (VISC) [mm²/s]")
    FLASH_in = input_val("Flash point (FLASH) [°C]")
    BP50_in = input_val("BP50 (T50) [°C]")
    FREEZE_in = input_val("Freeze point [°C]")
    TOTAL_in = input_val("Sulfur (TOTAL) [ppm]")

    st.markdown("---")
    st.markdown("**Model / UI Options**")
    use_imputer_draws = st.checkbox("Use multiple imputer draws (estimate uncertainty)", value=True)
    n_draws = st.slider("Imputer draws", min_value=1, max_value=200, value=50, step=1) if use_imputer_draws else 1
    show_rf = st.checkbox("Show RandomForest predictions", value=True)
    show_pls = st.checkbox("Show PLS predictions", value=True)
    safety_margin_cn = st.number_input("Safety margin for CN (units)", value=float(SAFETY_MARGINS["CN"]), step=0.5)
    # Button
    run_button = st.button("🔍 Predict properties")

# Load dataset & models (in main)
df = load_dataset()
# set medians for sidebar defaults if missing
meds = df.median()
# If user didn't input and we want to show convenient defaults in text boxes it's tricky; leave as is.

# Prepare provided dict from sidebar inputs
provided = {}
if CN_in is not None: provided["CN"] = CN_in
if D4052_in is not None: provided["D4052"] = D4052_in
if VISC_in is not None: provided["VISC"] = VISC_in
if FLASH_in is not None: provided["FLASH"] = FLASH_in
if BP50_in is not None: provided["BP50"] = BP50_in
if FREEZE_in is not None: provided["FREEZE"] = FREEZE_in
if TOTAL_in is not None: provided["TOTAL"] = TOTAL_in

# Validate enough inputs
if run_button:
    if len(provided) < 2:
        st.warning("Please provide at least two known properties to make reliable predictions.")
        st.stop()

    # 1) Ensure imputer exists or train
    imp, scaler_info = load_imputer()
    if imp is None or scaler_info is None:
        with st.spinner("Training imputer on dataset (learn relationships)..."):
            imp, scaler_info = train_imputer(df, sample_posterior=True, n_iter=30)
        st.success("Trained imputer and saved.")

    # 2) Produce imputer-based predictions (n_draws)
    with st.spinner("Running imputer to estimate missing properties..."):
        df_draws = imputer_transform_single(imp, scaler_info, provided, n_draws=n_draws)
    mean_imputed = df_draws.mean(axis=0)
    std_imputed = df_draws.std(axis=0)

    # 3) For predictive models (RF/PLS) we'll need model input columns: D4052,VISC,TOTAL,FLASH
    model_input_cols = ["D4052", "VISC", "TOTAL", "FLASH"]
    # If some not provided, we use imputer mean to fill them
    model_input = []
    for c in model_input_cols:
        if c in provided:
            model_input.append(float(provided[c]))
        else:
            # use imputer mean for the column
            model_input.append(float(mean_imputed[c]))

    # Train or load RF/PLS models for mapping inputs->outputs (outputs chosen: CN, BP50, FREEZE)
    output_cols = ["CN", "BP50", "FREEZE"]
    rf = pls = scaler_model = None
    if os.path.exists(RF_MODEL_PATH) and os.path.exists(PLS_MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            rf = joblib.load(RF_MODEL_PATH)
            pls = joblib.load(PLS_MODEL_PATH)
            scaler_model = joblib.load(SCALER_PATH)
        except Exception:
            rf = pls = scaler_model = None
    if rf is None or pls is None or scaler_model is None:
        with st.spinner("Training predictive models (RandomForest + PLS) from dataset..."):
            rf, pls, scaler_model, metrics = train_predict_models(df, model_input_cols, output_cols)
        st.success("Trained predictive models and saved.")
    # Prepare user input vector for predictive models
    X_user = np.array([model_input])
    X_user_s = scaler_model.transform(X_user)
    # RF predict and uncertainty
    rf_mean, rf_std = rf_predict_with_uncertainty(rf, X_user_s)
    rf_mean = rf_mean.flatten()
    rf_std = rf_std.flatten()
    # PLS predict
    pls_pred = pls.predict(X_user_s).flatten()

    # Build final combined vector:
    # We'll use imputer mean for all properties (safer), but for CN/BP50/FREEZE show RF/PLS predictions as well
    imputer_mean_dict = mean_imputed.to_dict()
    preds_combined = imputer_mean_dict.copy()
    # overwrite CN/BP50/FREEZE with RF mean (primary) and also keep PLS for comparison
    preds_combined["CN"] = float(rf_mean[0])
    preds_combined["BP50"] = float(rf_mean[1])
    preds_combined["FREEZE"] = float(rf_mean[2])

    # Compute quality index & pass/fail
    QUALITY_WEIGHTS_LOCAL = QUALITY_WEIGHTS.copy()
    SAFETY_MARGINS_LOCAL = SAFETY_MARGINS.copy()
    SAFETY_MARGINS_LOCAL["CN"] = float(safety_margin_cn)
    quality_score, pass_label, per_prop_scores = compute_quality_from_vector(preds_combined, weights=QUALITY_WEIGHTS_LOCAL,
                                                                             specs=SPEC_LIMITS, margins=SAFETY_MARGINS_LOCAL)

    # ---------------------------
    # Display results
    # ---------------------------
    st.markdown("## ✅ Results")
    st.markdown("<div class='card'><div style='display:flex; gap:20px;'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown("<div class='metric-box'><h4>Quality Score</h4><div class='big-number'>"
                    f"{quality_score} / 100</div><p class='small-muted'>Weighted index across properties</p></div>",
                    unsafe_allow_html=True)
    with col2:
        badge_color = "#27AE60" if pass_label=="PASS" else ("#E67E22" if pass_label=="CHECK" else "#C0392B")
        st.markdown(f"<div class='metric-box'><h4>Decision</h4>"
                    f"<div style='font-size:20px;padding:8px;border-radius:6px;background:{badge_color};color:#fff;text-align:center'>{pass_label}</div>"
                    f"<p class='small-muted'>PASS / CHECK / FAIL (conservative)</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-box'><h4>Source</h4>"
                    f"<p class='small-muted'>Imputer + RandomForest predictions (RF primary). PLS provided for comparison.</p></div>",
                    unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("### Predicted values (imputer mean ± std and RF/PLS where available)")
    table_rows = []
    for p in ALL_PROPERTIES:
        im_mean = mean_imputed[p]
        im_std = std_imputed[p]
        rf_val = ""
        rf_unc = ""
        pls_val = ""
        if p in output_cols:
            # RF outputs order maps to output_cols
            idx = output_cols.index(p)
            rf_val = f"{rf_mean[idx]:.3f}"
            rf_unc = f"±{rf_std[idx]:.3f}"
            pls_val = f"{pls_pred[idx]:.3f}"
        table_rows.append({
            "Property": p,
            "Imputer_mean": f"{im_mean:.3f}",
            "Imputer_std": f"{im_std:.3f}",
            "RF_pred": rf_val,
            "RF_std": rf_unc,
            "PLS_pred": pls_val
        })
    df_results = pd.DataFrame(table_rows).set_index("Property")
    st.table(df_results)

    # Show a small comparison plot for RF vs actual (if we have X_test from training - metrics available)
    st.markdown("### Quick model metrics (on held-out data when models were trained)")
    try:
        # If metrics available from last training, display; else compute quick metrics using train_predict_models returned metrics earlier
        if 'metrics' in locals():
            m = metrics
            outcols = m["output_cols"]
            rf_r2 = m["rf_r2"]
            pls_r2 = m["pls_r2"]
            metrics_df = pd.DataFrame({
                "Output": outcols,
                "RF_R2": [round(v,3) for v in rf_r2],
                "PLS_R2": [round(v,3) for v in pls_r2]
            })
            st.table(metrics_df.set_index("Output"))
    except Exception:
        pass

    # Downloadable CSV of predictions and input
    payload = {**provided}
    for k in ALL_PROPERTIES:
        payload[f"{k}_pred_imputer_mean"] = float(mean_imputed[k])
        payload[f"{k}_pred_imputer_std"] = float(std_imputed[k])
    for i,k in enumerate(output_cols):
        payload[f"{k}_pred_rf"] = float(rf_mean[i])
        payload[f"{k}_pred_rf_std"] = float(rf_std[i])
        payload[f"{k}_pred_pls"] = float(pls_pred[i])
    payload["quality_score"] = quality_score
    payload["decision"] = pass_label
    df_payload = pd.DataFrame([payload])

    st.download_button("⬇️ Download prediction CSV", df_payload.to_csv(index=False).encode('utf-8'), file_name="fuel_prediction.csv", mime="text/csv")

    # Optional plots: scatter density vs CN/BP50/FREEZE
    if st.checkbox("Show exploratory scatter plots (training data)"):
        fig, axs = plt.subplots(1, 3, figsize=(12,3))
        axs[0].scatter(df["D4052"], df["CN"], alpha=0.6)
        axs[0].set_xlabel("D4052"); axs[0].set_ylabel("CN")
        axs[1].scatter(df["D4052"], df["BP50"], alpha=0.6)
        axs[1].set_xlabel("D4052"); axs[1].set_ylabel("BP50")
        axs[2].scatter(df["D4052"], df["FREEZE"], alpha=0.6)
        axs[2].set_xlabel("D4052"); axs[2].set_ylabel("FREEZE")
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### Notes")
    st.markdown(
        """
        * Predictions are **screening-level**. Borderline or critical samples should be sent for laboratory verification.
        * The imputer learns multivariate relationships between properties — it will generate estimates for any missing properties.
        * The RandomForest mapping (D4052,VISC,TOTAL,FLASH → CN,BP50,FREEZE) is trained on complete rows only and can be more accurate for those targets.
        * Adjust the safety margin for CN and other properties if you want the auto-PASS to be more or less conservative.
        """
    )
    st.success("Prediction complete.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray'>Built with ❤️ — Streamlit • Fuel Quality Project</p>", unsafe_allow_html=True)
