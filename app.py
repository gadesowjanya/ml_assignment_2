from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Import metrics engines for Requirement (d)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MODEL_BUNDLE_PATH = ARTIFACT_DIR / "model_bundle.joblib"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.csv"

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🧪")
st.title("Breast Cancer Wisconsin (Diagnostic)")
st.write(
    "This app compares five classifiers and calculates real-time evaluation "
    "metrics on an uploaded test dataset containing true labels."
)

if not MODEL_BUNDLE_PATH.is_file() or not METRICS_PATH.is_file():
    st.error("Model artifacts are missing. Please verify artifacts/ paths.")
    st.stop()

# Load saved pipelines without retraining
bundle = joblib.load(MODEL_BUNDLE_PATH)
models = bundle["models"]
feature_names = bundle["feature_names"]
baseline_metrics = pd.read_csv(METRICS_PATH)

# Feature (c) - Display of baseline evaluation metrics
st.subheader("1. Baseline Model Comparison (Notebook Split)")
st.dataframe(baseline_metrics.round(3), hide_index=True, width="stretch")

# Feature (b) - Model selection dropdown
st.subheader("2. Run Evaluation on Custom Test Data")
selected_name = st.selectbox("Select model for live test evaluation", list(models.keys()))

# Updated Template Structure requiring Target Label
template = pd.DataFrame(columns=["Diagnosis"] + feature_names)
st.download_button(
    "Download Test CSV Template",
    data=template.to_csv(index=False),
    file_name="test_data_template.csv",
    mime="text/csv",
)

# Feature (a) - Dataset upload option (CSV)
uploaded_file = st.file_uploader("Upload your test dataset CSV", type="csv")
if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()

    # Verify Target Presence
    if "Diagnosis" not in uploaded_data.columns:
        st.error("CSV is missing the mandatory target column: 'Diagnosis'")
        st.stop()

    # Verify Feature Column Presence
    missing_columns = [name for name in feature_names if name not in uploaded_data.columns]
    if missing_columns:
        st.error("CSV is missing required feature columns: " + ", ".join(missing_columns))
        st.stop()

    # Extract target and features cleanly
    y_true = uploaded_data["Diagnosis"].astype(str).str.strip()
    X_test = uploaded_data.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")

    if X_test.isna().any().any():
        st.error("All required feature values must be numeric and non-empty.")
        st.stop()

    # Predict using the pre-loaded dictionary pipeline
    predictions = models[selected_name].predict(X_test)
    
    # Standardize arrays to string matching for comparison stability
    predictions = [str(p).strip() for p in predictions]

    # Feature (c) - Live Calculation of Evaluation Metrics on Uploaded Data
    st.subheader(f"Live Evaluation Metrics ({selected_name})")
    acc = accuracy_score(y_true, predictions)
    
    # Treat malignant (M) as positive class
    prec = precision_score(y_true, predictions, pos_label="M", zero_division=0)
    rec = recall_score(y_true, predictions, pos_label="M", zero_division=0)
    f1 = f1_score(y_true, predictions, pos_label="M", zero_division=0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision (M)", f"{prec:.3f}")
    col3.metric("Recall (M)", f"{rec:.3f}")
    col4.metric("F1-Score (M)", f"{f1:.3f}")

    # Feature (d) - Confusion Matrix & Classification Report
    st.subheader("Diagnostics Output")
    
    tab1, tab2 = st.tabs(["Confusion Matrix Table", "Detailed Classification Report"])
    
    with tab1:
        labels = sorted(list(set(y_true).union(set(predictions))))
        cm = confusion_matrix(y_true, predictions, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Predicted {l}" for l in labels])
        st.dataframe(cm_df, width="stretch")
        st.caption("Rows show actual distribution; columns map model assertions.")

    with tab2:
        report = classification_report(y_true, predictions, zero_division=0)
        st.code(report)
