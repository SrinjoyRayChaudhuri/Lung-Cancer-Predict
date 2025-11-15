# src/app.py
import os
import joblib
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ---------------- Paths ----------------
BASE = os.path.dirname(__file__)               # .../src
ROOT = os.path.dirname(BASE)
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
PIPE_PATH = os.path.join(MODELS_DIR, "lung_cancer_pipeline.pkl")
FEATURES_JSON = os.path.join(MODELS_DIR, "features.json")
CM_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")

# ---------------- Helpers ----------------
def abort(msg):
    st.error(msg)
    st.stop()

def load_pipeline(path):
    if not os.path.exists(path):
        return None, f"Pipeline file not found at: {path}"
    try:
        obj = joblib.load(path)
    except Exception as e:
        return None, f"Failed to joblib.load('{path}'): {e}"
    if not hasattr(obj, "predict"):
        return obj, "Loaded object does not implement .predict (it is not a model/pipeline)."
    return obj, None

def find_dataset_csv():
    if not os.path.isdir(DATA_DIR):
        return None
    candidates = ["lung_cancer_dataset.csv", "Lung_cancer_dataset.csv", "lung_cancer.csv"]
    for c in candidates:
        p = os.path.join(DATA_DIR, c)
        if os.path.exists(p):
            return p
    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(".csv"):
            return os.path.join(DATA_DIR, f)
    return None

def read_features_json():
    if os.path.exists(FEATURES_JSON):
        try:
            with open(FEATURES_JSON, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def infer_raw_input_columns(pipeline_obj):
    fj = read_features_json()
    if fj and isinstance(fj, dict) and "raw_input_columns" in fj:
        return list(fj["raw_input_columns"])
    try:
        preproc = pipeline_obj.named_steps.get("preprocessor", None)
        if preproc is not None and hasattr(preproc, "transformers_"):
            cols = []
            for name, trans, cols_spec in preproc.transformers_:
                if isinstance(cols_spec, (list, tuple, np.ndarray)):
                    cols.extend(list(cols_spec))
            if cols:
                return cols
    except Exception:
        pass
    return None

def get_classifier_from_pipeline(pipeline_obj):
    """Return the classifier step inside pipeline, or pipeline itself if it's the estimator."""
    try:
        # search named steps for an estimator that looks like classifier
        for name, step in getattr(pipeline_obj, "named_steps", {}).items():
            # step that has predict_proba or classes_ likely to be classifier
            if hasattr(step, "predict") and (hasattr(step, "predict_proba") or hasattr(step, "classes_")):
                return step
    except Exception:
        pass
    # fallback: pipeline may itself be final estimator (rare)
    if hasattr(pipeline_obj, "predict"):
        return pipeline_obj
    return None

# ---------------- Load pipeline ----------------
st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")
st.title("🫁 Lung Cancer Prediction")

pipeline, err = load_pipeline(PIPE_PATH)
if pipeline is None:
    abort(f"Cannot load model pipeline.\n{err}\n\nMake sure you ran src/create_model.py to produce {PIPE_PATH}")

if err is not None:
    st.error("Model load diagnostic:")
    st.write(err)
    st.write("Contents of models/:")
    try:
        st.write(sorted(os.listdir(MODELS_DIR)))
    except Exception:
        pass
    abort("Loaded object is not a usable model/pipeline. Re-run training script to produce a pipeline file.")

# ---------------- get expected raw input columns ----------------
raw_input_cols = infer_raw_input_columns(pipeline)
if not raw_input_cols:
    abort(
        "Could not determine input columns the pipeline expects.\n"
        "Ensure src/create_model.py saved models/features.json or pipeline has a 'preprocessor' with transformers_."
    )

st.markdown(f"**Model expects these input columns (first 20):** {raw_input_cols[:20]}")

# ---------------- Load dataset sample ----------------
dataset_path = find_dataset_csv()
dataset_sample = None
if dataset_path:
    try:
        dataset_sample = pd.read_csv(dataset_path, nrows=5000)
    except Exception:
        dataset_sample = None

# ---------------- Build UI ----------------
st.subheader("Enter patient details")

ui_values = {}
cols = st.columns(2)
for i, col in enumerate(raw_input_cols):
    label = col.replace("_", " ").title()

    # If dataset sample exists and includes column, use its unique values to create a selectbox
    choices = None
    if dataset_sample is not None and col in dataset_sample.columns:
        uniq = dataset_sample[col].dropna().unique()
        if 1 < len(uniq) <= 40:
            # Keep original string order but convert to readable strings
            choices = list(map(str, sorted(map(str, uniq))))
            # common heuristic make Yes/No order sensible
            if set([c.lower() for c in choices]) >= {"yes", "no"}:
                choices = ["No", "Yes"]

    # Heuristics by column name
    if choices is None:
        if "age" in col.lower():
            ui_values[col] = cols[i % 2].number_input(label, min_value=0, max_value=120, value=40)
            continue
        if "blood" in col.lower():
            ui_values[col] = cols[i % 2].selectbox(label, ["A", "B", "AB", "O"])
            continue
        if "gender" in col.lower():
            ui_values[col] = cols[i % 2].selectbox(label, ["Male", "Female"])
            continue
        if any(k in col.lower() for k in ("smok","alcoho","fatigue","cough","chest","shortness","wheez","allerg","chronic","passive","yellow","peer","press","obes")):
            ui_values[col] = cols[i % 2].selectbox(label, ["Yes", "No"])
            continue
        # fallback text input
        ui_values[col] = cols[i % 2].text_input(label, value="")
    else:
        ui_values[col] = cols[i % 2].selectbox(label, choices)

st.markdown("**Entered values (preview):**")
st.write(ui_values)

# ---------------- Predict ----------------
if st.button("Predict"):
    try:
        # Build DataFrame matching the raw_input_cols order
        row = {c: ui_values.get(c, "") for c in raw_input_cols}
        X = pd.DataFrame([row], columns=raw_input_cols)

        # pipeline will do preprocessing + predict
        probs = pipeline.predict_proba(X)[0] if hasattr(pipeline, "predict_proba") else None
        pred = pipeline.predict(X)[0]

        # robust way to get classes as a plain list
        clf = get_classifier_from_pipeline(pipeline)
        classes_attr = None
        if clf is not None and hasattr(clf, "classes_"):
            classes_attr = getattr(clf, "classes_")
        elif hasattr(pipeline, "classes_"):
            classes_attr = getattr(pipeline, "classes_")
        else:
            classes_attr = None

        classes = list(classes_attr) if classes_attr is not None else []

        # Display probabilities in a safe way
        if probs is not None:
            st.subheader("Predicted probabilities")
            if classes and len(classes) == len(probs):
                for cls, p in zip(classes, probs):
                    st.write(f"- **{cls}**: {p*100:.2f}%")
                final_label = classes[int(np.argmax(probs))]
            else:
                # fallback: show index-based probs
                for i, p in enumerate(probs):
                    st.write(f"- class_{i}: {p*100:.2f}%")
                final_label = pred
        else:
            final_label = pred

        st.markdown("### Final prediction")
        st.success(f"Predicted Lung Cancer Risk: **{final_label}**")

    except Exception as e:
        # give diagnostic info to debug quickly
        st.exception(f"Prediction failed: {e}\n\nDiagnostics:\n- pipeline type: {type(pipeline)}\n- models/ files: {sorted(os.listdir(MODELS_DIR))}")
        raise

# ---------------- Confusion matrix / performance ----------------
st.markdown("---")
st.subheader("Model performance / Confusion matrix")

if st.button("Show Confusion Matrix"):
    if os.path.exists(CM_PATH):
        st.image(CM_PATH, use_column_width=True)
    else:
        if not dataset_path:
            st.info("No saved confusion matrix image and no dataset in data/ to compute it.")
        else:
            try:
                df_all = pd.read_csv(dataset_path)
                target_candidates = ["Lung Cancer Risk", "LUNG CANCER RISK", "Lung_Cancer_Risk", "Level", "LUNG_CANCER", "Lung Cancer"]
                target_col = None
                for t in target_candidates:
                    if t in df_all.columns:
                        target_col = t
                        break
                if target_col is None:
                    target_col = df_all.columns[-1]

                X_all = df_all[raw_input_cols]
                y_all = df_all[target_col].astype(str)

                y_pred = pipeline.predict(X_all)

                labels = np.unique(y_all)
                cm = confusion_matrix(y_all, y_pred, labels=labels)

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                st.write("### Classification report")
                report = classification_report(y_all, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

            except Exception as e:
                st.exception(f"Could not compute confusion matrix: {e}")

# ---------------- Debug expander ----------------
with st.expander("Debug info"):
    st.write("models dir contents:", sorted(os.listdir(MODELS_DIR)) if os.path.isdir(MODELS_DIR) else "models/ not found")
    st.write("pipeline type:", type(pipeline))
    try:
        st.write("pipeline steps:", list(getattr(pipeline, "steps", [])))
    except Exception:
        pass
    st.write("dataset preview path:", dataset_path if dataset_path else "no dataset")
    if dataset_path:
        try:
            st.dataframe(pd.read_csv(dataset_path, nrows=3))
        except Exception:
            st.write("Could not preview dataset.")
