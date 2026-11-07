from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MODEL_BUNDLE_PATH = ARTIFACT_DIR / "model_bundle.joblib"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.csv"

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🧪")
st.title("Breast Cancer Wisconsin (Diagnostic)")
st.write(
    "This educational app compares five classifiers and applies a selected "
    "trained model to rows of dataset features."
)
st.warning(
    "For educational use only. This model is not a medical device and must not "
    "be used for diagnosis or treatment decisions."
)

if not MODEL_BUNDLE_PATH.is_file() or not METRICS_PATH.is_file():
    st.error(
        "Model artifacts are missing. Run the final save-artifacts cell in "
        "models.ipynb first; it should create artifacts/model_bundle.joblib "
        "and artifacts/model_metrics.csv."
    )
    st.stop()

bundle = joblib.load(MODEL_BUNDLE_PATH)
models = bundle["models"]
feature_names = bundle["feature_names"]
metrics = pd.read_csv(METRICS_PATH)

st.subheader("Model comparison")
st.caption("Metrics are from the notebook's held-out test split.")
st.dataframe(metrics.round(3), hide_index=True, width="stretch")

st.subheader("Make predictions")
selected_name = st.selectbox("Choose a model", list(models.keys()))

template = pd.DataFrame(columns=feature_names)
st.download_button(
    "Download feature CSV template",
    data=template.to_csv(index=False),
    file_name="prediction_template.csv",
    mime="text/csv",
)
st.caption(
    "Fill the template with one sample per row. Keep the feature column names "
    "unchanged and provide numeric values."
)

uploaded_file = st.file_uploader("Upload a feature CSV", type="csv")
if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()

    missing_columns = [name for name in feature_names if name not in uploaded_data.columns]
    if missing_columns:
        st.error("CSV is missing required feature columns: " + ", ".join(missing_columns))
        st.stop()

    features = uploaded_data.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        st.error("All required feature values must be numeric and non-empty.")
        st.stop()

    if not (features.abs() != float("inf")).all().all():
        st.error("Feature values must be finite numbers.")
        st.stop()

    predictions = models[selected_name].predict(features)
    prediction_labels = pd.Series(predictions).map(
        {"B": "Benign", "M": "Malignant"}
    ).fillna(pd.Series(predictions).astype(str))

    results = pd.DataFrame(
        {
            "Prediction": prediction_labels,
            "Model label": predictions,
        },
        index=uploaded_data.index,
    )
    st.dataframe(results, width="stretch")