"""
SmartGrid-AIDM — Adversarially Robust Intrusion Detection for Smart Grids
Streamlit dashboard for model evaluation, dataset exploration, and attack analysis.

Run:
    streamlit run app.py

Expected project layout:
    app.py
    dataset/
    models/
        balanced_unified_random_forest.pkl
        unified_no_timediff_experimental.pkl
    src/  (optional training/evaluation scripts)

Important:
- Metrics are computed from the data and models available at runtime.
- The recorded historical metrics are clearly labelled as historical reference values.
- Synthetic subtle-attack results are reported separately from ordinary test accuracy.
- "Model probability" means predict_proba output and may not be calibrated confidence.
"""

from __future__ import annotations

import io
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# 1. Application configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SmartGrid-AIDM | Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "outputs"
FEATURES = [
    "Actual frequency value",
    "Fraction of second",
    "PMU time",
    "PDC time",
    "interarrival time",
    "time difference",
]
FIVE_FEATURES = [name for name in FEATURES if name != "time difference"]
LABEL_COLUMN = "attack_type"
CLASS_ORDER = ["Normal", "FDI", "TSA"]

MODEL_FILES = {
    "Six-feature Random Forest": MODEL_DIR / "balanced_unified_random_forest.pkl",
    "Five-feature Random Forest": MODEL_DIR / "unified_no_timediff_experimental.pkl",
}

DATA_FILES = {
    "FDI training / evaluation data": DATASET_DIR / "processed_fdi.csv",
    "TSA training / evaluation data": DATASET_DIR / "processed_tsa.csv",
    "FDI labelled test data": DATASET_DIR / "processed_fdi_test.csv",
    "TSA labelled test data": DATASET_DIR / "processed_tsa_test.csv",
    "Subtle FDI test data": DATASET_DIR / "final_subtle_fdi_test.csv",
    "Subtle TSA test data": DATASET_DIR / "final_subtle_tsa_test.csv",
}

# Historical values from earlier runs. These are displayed as references only,
# and are not silently substituted for metrics computed in the current session.
HISTORICAL_METRICS = {
    "Six-feature Random Forest": {
        "accuracy": 87.61,
        "normal_recall": 68.54,
        "fdi_recall": 94.30,
        "tsa_recall": 100.00,
        "normal_false_alarm": 31.46,
        "matrix": [[3427, 1573, 0], [285, 4715, 0], [0, 0, 5000]],
        "description": "Historical balanced test of 15,000 rows.",
    },
    "Five-feature Random Forest": {
        "accuracy": 97.24,
        "normal_recall": 97.76,
        "fdi_recall": 94.28,
        "tsa_recall": 99.68,
        "normal_false_alarm": 2.24,
        "matrix": [[4888, 101, 11], [286, 4714, 0], [0, 16, 4984]],
        "description": "Historical balanced test of 15,000 rows.",
    },
}
HISTORICAL_SUBTLE = {
    "Subtle FDI": {
        "detected": 1657,
        "total": 15436,
        "detection_rate": 10.73,
        "source": "Historical five-feature model run",
    },
    "Subtle TSA": {
        "detected": 15427,
        "total": 15436,
        "detection_rate": 99.94,
        "source": "Historical five-feature model run",
    },
}


# ---------------------------------------------------------------------------
# 2. General utility functions
# ---------------------------------------------------------------------------

def file_exists(path: Path) -> bool:
    """Return whether a path exists and is a regular file."""
    return path.exists() and path.is_file()


def safe_read_csv(path: Path, **kwargs: Any) -> Optional[pd.DataFrame]:
    """Read a CSV without stopping the whole dashboard if one file is absent."""
    if not file_exists(path):
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        st.warning(f"Could not read `{path.name}`: {exc}")
        return None


@st.cache_data(show_spinner=False)
def cached_csv(path_string: str, modified_time: float) -> Optional[pd.DataFrame]:
    """Cache a CSV using its modified time as part of the cache key."""
    path = Path(path_string)
    if not file_exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    """Cached wrapper for CSV loading."""
    if not file_exists(path):
        return None
    return cached_csv(str(path), path.stat().st_mtime)


@st.cache_resource(show_spinner="Loading available models...")
def load_models() -> Dict[str, Any]:
    """Load whichever model files are available."""
    loaded: Dict[str, Any] = {}
    for name, path in MODEL_FILES.items():
        if file_exists(path):
            try:
                loaded[name] = joblib.load(path)
            except Exception as exc:
                loaded[name] = exc
    return loaded


def available_model_names(models: Dict[str, Any]) -> List[str]:
    """Return only models that loaded successfully."""
    return [
        name for name, model in models.items()
        if not isinstance(model, Exception) and hasattr(model, "predict")
    ]


def normalize_label(value: Any) -> str:
    """Normalize common class label spellings."""
    text = str(value).strip()
    upper = text.upper()
    mapping = {
        "NORMAL": "Normal",
        "BENIGN": "Normal",
        "0": "Normal",
        "FDI": "FDI",
        "FALSE DATA INJECTION": "FDI",
        "FALSE_DATA_INJECTION": "FDI",
        "1": "FDI",
        "TSA": "TSA",
        "TIME SYNCHRONIZATION ATTACK": "TSA",
        "TIME_SYNCHRONIZATION_ATTACK": "TSA",
        "2": "TSA",
    }
    return mapping.get(upper, text)


def normalize_labels(series: pd.Series) -> pd.Series:
    """Normalize all labels in a Series."""
    return series.astype(str).map(normalize_label)


def model_feature_names(model: Any) -> List[str]:
    """Use model metadata if available; otherwise infer the known feature set."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(item) for item in names]
    count = getattr(model, "n_features_in_", None)
    if count == 5:
        return FIVE_FEATURES
    return FEATURES


def prepare_features(frame: pd.DataFrame, model: Any) -> Tuple[pd.DataFrame, List[str]]:
    """Return the feature frame in the exact order expected by the model."""
    wanted = model_feature_names(model)
    missing = [column for column in wanted if column not in frame.columns]
    if missing:
        raise ValueError("Missing required model features: " + ", ".join(missing))
    features = frame[wanted].copy()
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, wanted


def model_predict(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Run prediction with basic numeric validation."""
    features, _ = prepare_features(frame, model)
    if features.isna().any().any():
        features = features.fillna(features.median(numeric_only=True)).fillna(0)
    return np.asarray(model.predict(features)).astype(str)


def model_probabilities(model: Any, frame: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Return predict_proba values if supported."""
    if not hasattr(model, "predict_proba"):
        return None, None
    features, _ = prepare_features(frame, model)
    if features.isna().any().any():
        features = features.fillna(features.median(numeric_only=True)).fillna(0)
    probabilities = np.asarray(model.predict_proba(features))
    classes = [normalize_label(item) for item in getattr(model, "classes_", range(probabilities.shape[1]))]
    return probabilities, classes


def percentage(value: Any, digits: int = 2) -> str:
    """Format a ratio or percentage value."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}%"


def safe_divide(numerator: float, denominator: float) -> float:
    """Division with zero-denominator protection."""
    return float(numerator / denominator) if denominator else 0.0


def dataframe_download(frame: pd.DataFrame, filename: str, label: str) -> None:
    """Add a CSV download button."""
    payload = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=payload,
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


def metric_card(label: str, value: Any, help_text: Optional[str] = None) -> None:
    """Render a compact metric card."""
    st.metric(label=label, value=value, help=help_text)


def section_title(title: str, description: Optional[str] = None) -> None:
    """Consistent section heading."""
    st.subheader(title)
    if description:
        st.caption(description)



def display_table(frame: pd.DataFrame, key: Optional[str] = None, height: Optional[int] = None) -> None:
    """Render a dataframe with a download option."""
    if height is None:
        st.dataframe(frame, use_container_width=True, hide_index=True)
    else:
        st.dataframe(frame, use_container_width=True, height=height, hide_index=True)

    filename = (key or "smartgrid_table").lower().replace(" ", "_") + ".csv"
    dataframe_download(frame, filename, "Download table as CSV")


def find_class_name(classes: Sequence[str], wanted: str) -> Optional[str]:
    """Find a class in a list using normalized spelling."""
    normalized = {normalize_label(item): item for item in classes}
    return normalized.get(wanted)


# ---------------------------------------------------------------------------
# 3. Dataset and model evaluation functions
# ---------------------------------------------------------------------------

def dataset_summary(frame: pd.DataFrame, label: str) -> Dict[str, Any]:
    """Summarize dataset size, labels, missing values, and feature count."""
    label_counts: Dict[str, int] = {}
    if LABEL_COLUMN in frame.columns:
        counts = normalize_labels(frame[LABEL_COLUMN]).value_counts()
        label_counts = {str(k): int(v) for k, v in counts.items()}
    return {
        "Dataset": label,
        "Rows": int(frame.shape[0]),
        "Columns": int(frame.shape[1]),
        "Feature columns present": int(sum(col in frame.columns for col in FEATURES)),
        "Missing cells": int(frame.isna().sum().sum()),
        "Duplicate rows": int(frame.duplicated().sum()),
        "Labelled": LABEL_COLUMN in frame.columns,
        "Class counts": json.dumps(label_counts, ensure_ascii=False),
    }


def binary_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class: str,
    all_classes: Sequence[str],
) -> Dict[str, float]:
    """Calculate one-vs-rest binary metrics for a selected class."""
    true_positive = (y_true == positive_class)
    pred_positive = (y_pred == positive_class)
    tp = int(np.sum(true_positive & pred_positive))
    fp = int(np.sum(~true_positive & pred_positive))
    fn = int(np.sum(true_positive & ~pred_positive))
    tn = int(np.sum(~true_positive & ~pred_positive))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    fpr = safe_divide(fp, fp + tn)
    fnr = safe_divide(fn, fn + tp)
    npv = safe_divide(tn, tn + fn)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision (%)": precision * 100,
        "Recall / sensitivity (%)": recall * 100,
        "Specificity (%)": specificity * 100,
        "F1-score (%)": f1 * 100,
        "False-positive rate (%)": fpr * 100,
        "False-negative rate (%)": fnr * 100,
        "Negative predictive value (%)": npv * 100,
        "Support": int(np.sum(true_positive)),
    }


def evaluate_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    model_name: str,
) -> Dict[str, Any]:
    """Calculate overall and per-class classification metrics."""
    truth = np.asarray([normalize_label(item) for item in y_true], dtype=object)
    pred = np.asarray([normalize_label(item) for item in y_pred], dtype=object)
    classes = [name for name in CLASS_ORDER if name in set(truth) | set(pred)]
    if not classes:
        classes = sorted(set(truth) | set(pred))
    matrix = confusion_matrix(truth, pred, labels=classes)
    report = classification_report(
        truth,
        pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    p, r, f, support = precision_recall_fscore_support(
        truth, pred, labels=classes, zero_division=0
    )
    per_class: List[Dict[str, Any]] = []
    for index, class_name in enumerate(classes):
        details = binary_class_metrics(truth, pred, class_name, classes)
        per_class.append({
            "Model": model_name,
            "Class": class_name,
            "Precision (%)": p[index] * 100,
            "Recall (%)": r[index] * 100,
            "F1-score (%)": f[index] * 100,
            "Support": int(support[index]),
            "Specificity (%)": details["Specificity (%)"],
            "False-positive rate (%)": details["False-positive rate (%)"],
            "False-negative rate (%)": details["False-negative rate (%)"],
            "TP": details["TP"],
            "FP": details["FP"],
            "FN": details["FN"],
            "TN": details["TN"],
        })
    return {
        "model": model_name,
        "y_true": truth,
        "y_pred": pred,
        "classes": classes,
        "matrix": matrix,
        "accuracy": accuracy_score(truth, pred),
        "balanced_accuracy": balanced_accuracy_score(truth, pred),
        "precision_macro": precision_score(truth, pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(truth, pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(truth, pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(truth, pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(truth, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(truth, pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(truth, pred),
        "per_class": pd.DataFrame(per_class),
        "report": report,
    }


def build_overall_metrics(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten overall evaluation into a dashboard row."""
    return {
        "Model": evaluation["model"],
        "Samples evaluated": len(evaluation["y_true"]),
        "Accuracy (%)": evaluation["accuracy"] * 100,
        "Balanced accuracy (%)": evaluation["balanced_accuracy"] * 100,
        "Macro precision (%)": evaluation["precision_macro"] * 100,
        "Weighted precision (%)": evaluation["precision_weighted"] * 100,
        "Macro recall (%)": evaluation["recall_macro"] * 100,
        "Weighted recall (%)": evaluation["recall_weighted"] * 100,
        "Macro F1 (%)": evaluation["f1_macro"] * 100,
        "Weighted F1 (%)": evaluation["f1_weighted"] * 100,
        "Matthews correlation coefficient": evaluation["mcc"],
    }


def evaluate_frame(
    model: Any,
    frame: pd.DataFrame,
    model_name: str,
) -> Dict[str, Any]:
    """Evaluate a model on a labelled frame."""
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(f"Dataset must contain a `{LABEL_COLUMN}` column.")
    usable = frame.copy()
    usable[LABEL_COLUMN] = normalize_labels(usable[LABEL_COLUMN])
    usable = usable[usable[LABEL_COLUMN].isin(CLASS_ORDER)]
    if usable.empty:
        raise ValueError("No recognized Normal, FDI, or TSA labels were found.")
    predictions = model_predict(model, usable)
    return evaluate_predictions(usable[LABEL_COLUMN].to_numpy(), predictions, model_name)


def plot_confusion_matrix(evaluation: Dict[str, Any], normalize: bool = False) -> plt.Figure:
    """Create a confusion matrix figure."""
    matrix = evaluation["matrix"].astype(float)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0) * 100
    fig, ax = plt.subplots(figsize=(6, 4.5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(evaluation["classes"])),
        yticks=np.arange(len(evaluation["classes"])),
        xticklabels=evaluation["classes"],
        yticklabels=evaluation["classes"],
        xlabel="Predicted class",
        ylabel="Actual class",
        title=f"{evaluation['model']} — confusion matrix" + (" (%)" if normalize else ""),
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = f"{matrix[i, j]:.1f}%" if normalize else f"{int(matrix[i, j])}"
            ax.text(
                j, i, value, ha="center", va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )
    fig.tight_layout()
    return fig


def plot_class_metric(evaluations: Sequence[Dict[str, Any]], metric: str) -> plt.Figure:
    """Plot class-level metrics for each model."""
    rows = []
    for evaluation in evaluations:
        frame = evaluation["per_class"]
        for _, row in frame.iterrows():
            rows.append({
                "Model": evaluation["model"],
                "Class": row["Class"],
                "Value": row[metric],
            })
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not data.empty:
        pivot = data.pivot(index="Class", columns="Model", values="Value").fillna(0)
        pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(metric)
    ax.set_title(f"Per-class {metric}")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_metric_comparison(overall: pd.DataFrame, metrics: Sequence[str]) -> plt.Figure:
    """Plot multiple overall metric comparisons."""
    usable = overall[["Model", *metrics]].set_index("Model")
    fig, ax = plt.subplots(figsize=(10, 5))
    usable.plot(kind="bar", ax=ax)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Overall model metric comparison")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_class_distribution(frame: pd.DataFrame, title: str) -> plt.Figure:
    """Plot class distribution."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if LABEL_COLUMN not in frame.columns:
        ax.text(0.5, 0.5, "No attack_type label column", ha="center", va="center")
        ax.axis("off")
    else:
        counts = normalize_labels(frame[LABEL_COLUMN]).value_counts()
        counts.reindex(CLASS_ORDER, fill_value=0).plot(kind="bar", ax=ax)
        ax.set_ylabel("Rows")
        ax.set_xlabel("Class")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def plot_feature_histogram(frame: pd.DataFrame, feature: str) -> plt.Figure:
    """Plot a selected feature by class."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if feature not in frame.columns:
        ax.text(0.5, 0.5, f"{feature} is not available", ha="center", va="center")
        ax.axis("off")
    elif LABEL_COLUMN not in frame.columns:
        values = pd.to_numeric(frame[feature], errors="coerce").dropna()
        ax.hist(values, bins=40, alpha=0.8)
        ax.set_title(f"Distribution of {feature}")
    else:
        for class_name in CLASS_ORDER:
            values = pd.to_numeric(
                frame.loc[normalize_labels(frame[LABEL_COLUMN]) == class_name, feature],
                errors="coerce",
            ).dropna()
            if not values.empty:
                ax.hist(values, bins=40, alpha=0.45, label=class_name)
        ax.legend()
        ax.set_title(f"{feature} distribution by class")
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_correlation(frame: pd.DataFrame) -> plt.Figure:
    """Plot numeric-feature correlation matrix."""
    columns = [column for column in FEATURES if column in frame.columns]
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(columns) < 2:
        ax.text(0.5, 0.5, "At least two numeric features are needed", ha="center", va="center")
        ax.axis("off")
    else:
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
        corr = numeric.corr()
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(columns)), labels=columns, rotation=45, ha="right")
        ax.set_yticks(range(len(columns)), labels=columns)
        for i in range(len(columns)):
            for j in range(len(columns)):
                if pd.notna(corr.iloc[i, j]):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    return fig


def plot_prediction_distribution(frame: pd.DataFrame, prediction: np.ndarray, title: str) -> plt.Figure:
    """Plot model-predicted class distribution."""
    fig, ax = plt.subplots(figsize=(7, 4))
    series = pd.Series([normalize_label(item) for item in prediction]).value_counts()
    series.reindex(CLASS_ORDER, fill_value=0).plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Predicted rows")
    ax.set_xlabel("Predicted class")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Sidebar, initial loading, and common context
# ---------------------------------------------------------------------------

st.title("🛡️ SmartGrid-AIDM")
st.markdown(
    "**Adversarially Robust Intrusion Detection Mechanism for Smart Grid Security "
    "Using PMU Data**"
)
st.caption(
    "Explore the datasets, compare two Random Forest models, inspect class-level "
    "metrics, and test labelled or synthetic attack samples."
)

with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Choose a section",
        [
            "Project Overview",
            "Dataset Explorer",
            "Model Comparison",
            "Confusion Matrix",
            "Attack Simulator",
            "Subtle Attack Analysis",
            "Upload & Evaluate",
            "Feature Analysis",
            "Technical Details",
        ],
    )
    st.divider()
    st.write("**Project files**")
    st.caption(f"Root: `{ROOT}`")
    st.caption(f"Dataset folder: {'found' if DATASET_DIR.exists() else 'missing'}")
    st.caption(f"Model folder: {'found' if MODEL_DIR.exists() else 'missing'}")
    if st.button("Clear cached data and models"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

models_loaded = load_models()
models = {
    name: model for name, model in models_loaded.items()
    if not isinstance(model, Exception) and hasattr(model, "predict")
}
model_errors = {
    name: str(model) for name, model in models_loaded.items()
    if isinstance(model, Exception)
}
available_models = available_model_names(models_loaded)

datasets: Dict[str, pd.DataFrame] = {}
for label, path in DATA_FILES.items():
    loaded = load_csv(path)
    if loaded is not None:
        datasets[label] = loaded

if not models:
    st.error("No model loaded successfully. Check the files inside the `models` folder.")
    for name, error in model_errors.items():
        st.write(f"**{name}:** {error}")
    st.info(
        "Expected model files are `models/balanced_unified_random_forest.pkl` and "
        "`models/unified_no_timediff_experimental.pkl`."
    )

# Prefer the combined labelled dataset for general exploration when available.
default_dataset_name = next(
    (name for name in ["TSA training / evaluation data", "FDI training / evaluation data"] if name in datasets),
    next(iter(datasets), None),
)
default_dataset = datasets.get(default_dataset_name) if default_dataset_name else None


# ---------------------------------------------------------------------------
# 5. Page: Project overview
# ---------------------------------------------------------------------------

if page == "Project Overview":
    st.header("Project overview")
    st.write(
        "The dashboard evaluates machine-learning models that classify PMU telemetry "
        "as Normal, False Data Injection (FDI), or Time Synchronization Attack (TSA). "
        "It separates ordinary labelled-test performance from synthetic subtle-attack "
        "experiments, because these answer different questions."
    )

    total_dataset_rows = sum(len(frame) for frame in datasets.values())
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Models loaded", len(models))
    with col2:
        metric_card("CSV datasets found", len(datasets))
    with col3:
        metric_card("Rows across found CSVs", f"{total_dataset_rows:,}")
    with col4:
        metric_card("Known input features", len(FEATURES))

    section_title("What the dashboard evaluates")
    overview = pd.DataFrame([
        {"Area": "Overall classification", "Purpose": "Accuracy, balanced accuracy, macro/weighted precision, recall, F1, MCC"},
        {"Area": "Per-class behaviour", "Purpose": "Normal, FDI, and TSA precision, recall, specificity, F1, and support"},
        {"Area": "Error analysis", "Purpose": "Confusion matrix, false alarms, missed attacks, and class confusion"},
        {"Area": "Subtle attacks", "Purpose": "Detection rate on synthetic subtle FDI/TSA data, reported separately"},
        {"Area": "Feature analysis", "Purpose": "Distributions, missing values, correlations, and model feature importance when available"},
        {"Area": "Reproducibility", "Purpose": "Upload a labelled CSV to calculate fresh metrics from the loaded models"},
    ])
    display_table(overview, "overview_areas")

    section_title("Model availability")
    status_rows = []
    for name, path in MODEL_FILES.items():
        loaded = name in models
        model = models.get(name)
        status_rows.append({
            "Model": name,
            "File": str(path.relative_to(ROOT)),
            "File exists": file_exists(path),
            "Loaded": loaded,
            "Expected features": ", ".join(model_feature_names(model)) if loaded else "N/A",
            "Feature count": getattr(model, "n_features_in_", "N/A") if loaded else "N/A",
        })
    display_table(pd.DataFrame(status_rows), "model_availability")

    section_title("Read this before interpreting metrics")
    st.warning(
        "Accuracy can look high even when a model misses an important attack class. "
        "Always inspect per-class recall, false-positive rate, and the confusion matrix."
    )
    st.info(
        "The historical 97.24% overall accuracy recorded for the five-feature model "
        "does not mean it detects 97.24% of subtle FDI attacks. The subtle-FDI experiment "
        "has its own detection rate and must be reported separately."
    )

    section_title("Data files found")
    file_rows = []
    for label, path in DATA_FILES.items():
        frame = datasets.get(label)
        file_rows.append({
            "Dataset": label,
            "Path": str(path.relative_to(ROOT)),
            "Exists": file_exists(path),
            "Rows": len(frame) if frame is not None else 0,
            "Columns": frame.shape[1] if frame is not None else 0,
        })
    display_table(pd.DataFrame(file_rows), "dataset_file_status")


# ---------------------------------------------------------------------------
# 6. Page: Dataset explorer
# ---------------------------------------------------------------------------

elif page == "Dataset Explorer":
    st.header("Dataset explorer")
    if not datasets:
        st.error("No CSV datasets were found in the `dataset` folder.")
        st.stop()

    selected_name = st.selectbox("Select dataset", list(datasets.keys()))
    frame = datasets[selected_name].copy()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Rows", f"{len(frame):,}")
    with col2:
        metric_card("Columns", frame.shape[1])
    with col3:
        metric_card("Missing cells", f"{int(frame.isna().sum().sum()):,}")
    with col4:
        metric_card("Duplicate rows", f"{int(frame.duplicated().sum()):,}")

    section_title("Table 1 — Dataset summary")
    display_table(pd.DataFrame([dataset_summary(frame, selected_name)]), "dataset_summary")

    section_title("Table 2 — Class distribution")
    if LABEL_COLUMN in frame.columns:
        class_counts = normalize_labels(frame[LABEL_COLUMN]).value_counts().reindex(CLASS_ORDER, fill_value=0)
        distribution = pd.DataFrame({
            "Class": class_counts.index,
            "Rows": class_counts.values,
            "Percentage (%)": class_counts.values / max(len(frame), 1) * 100,
        })
        display_table(distribution, "class_distribution")
        st.pyplot(plot_class_distribution(frame, f"Class distribution — {selected_name}"), use_container_width=True)
    else:
        st.info("This dataset has no `attack_type` column.")

    section_title("Table 3 — Feature data quality")
    quality_rows = []
    for column in frame.columns:
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        quality_rows.append({
            "Column": column,
            "Data type": str(series.dtype),
            "Non-null": int(series.notna().sum()),
            "Missing": int(series.isna().sum()),
            "Unique values": int(series.nunique(dropna=True)),
            "Numeric values parsed": int(numeric.notna().sum()),
            "Mean": numeric.mean() if numeric.notna().any() else np.nan,
            "Std. deviation": numeric.std() if numeric.notna().any() else np.nan,
            "Minimum": numeric.min() if numeric.notna().any() else np.nan,
            "Maximum": numeric.max() if numeric.notna().any() else np.nan,
        })
    display_table(pd.DataFrame(quality_rows), "feature_quality")

    section_title("Preview and filtering")
    search_column = st.selectbox("Filter by column", ["(no filter)"] + list(frame.columns))
    filtered = frame
    if search_column != "(no filter)":
        query = st.text_input("Value contains", "")
        if query:
            filtered = frame[frame[search_column].astype(str).str.contains(query, case=False, na=False)]
    row_limit = st.slider("Rows to display", 5, 100, 20, step=5)
    st.dataframe(filtered.head(row_limit), use_container_width=True, hide_index=True)
    dataframe_download(filtered, "selected_dataset.csv", "Download filtered dataset")

    section_title("Feature distribution")
    numeric_features = [column for column in FEATURES if column in frame.columns]
    if numeric_features:
        selected_feature = st.selectbox("Feature to inspect", numeric_features)
        st.pyplot(plot_feature_histogram(frame, selected_feature), use_container_width=True)
    else:
        st.info("No known feature columns are present.")

    section_title("Feature correlation")
    st.pyplot(plot_correlation(frame), use_container_width=True)


# ---------------------------------------------------------------------------
# 7. Page: Model comparison
# ---------------------------------------------------------------------------

elif page == "Model Comparison":
    st.header("Model comparison")
    st.write(
        "This page displays the historical reference scores and, when a labelled "
        "dataset is selected, freshly calculated metrics for each model that can run."
    )

    historical_rows = []
    for name, values in HISTORICAL_METRICS.items():
        historical_rows.append({
            "Model": name,
            "Historical accuracy (%)": values["accuracy"],
            "Normal recall (%)": values["normal_recall"],
            "FDI recall (%)": values["fdi_recall"],
            "TSA recall (%)": values["tsa_recall"],
            "Normal false alarm (%)": values["normal_false_alarm"],
            "Test size": 15000,
            "Source": values["description"],
        })
    historical_frame = pd.DataFrame(historical_rows)
    section_title("Table 4 — Historical reference metrics")
    display_table(historical_frame, "historical_model_metrics")

    st.pyplot(
        plot_metric_comparison(
            historical_frame.rename(columns={
                "Historical accuracy (%)": "Accuracy (%)",
                "Normal recall (%)": "Normal recall (%)",
                "FDI recall (%)": "FDI recall (%)",
                "TSA recall (%)": "TSA recall (%)",
            }),
            ["Accuracy (%)", "Normal recall (%)", "FDI recall (%)", "TSA recall (%)"],
        ),
        use_container_width=True,
    )

    st.markdown("### Re-evaluate models on an available labelled CSV")
    labelled_names = [name for name, frame in datasets.items() if LABEL_COLUMN in frame.columns]
    if not labelled_names:
        st.warning("No labelled CSV was found. Use Upload & Evaluate to upload a labelled test CSV.")
    elif not models:
        st.warning("Models are not loaded, so fresh predictions cannot be calculated.")
    else:
        selected_data_name = st.selectbox("Evaluation dataset", labelled_names, key="comparison_dataset")
        eval_frame = datasets[selected_data_name]
        max_rows = min(len(eval_frame), 50000)
        eval_count = st.slider("Maximum rows for this evaluation", 100, max(100, max_rows), min(max_rows, 15000), step=100)
        if st.button("Calculate fresh comparison", type="primary"):
            sample = eval_frame.sample(n=min(eval_count, len(eval_frame)), random_state=42) if len(eval_frame) > eval_count else eval_frame
            fresh = []
            evaluations = []
            for name, model in models.items():
                try:
                    result = evaluate_frame(model, sample, name)
                    evaluations.append(result)
                    fresh.append(build_overall_metrics(result))
                except Exception as exc:
                    st.error(f"{name}: evaluation failed — {exc}")
            if fresh:
                fresh_frame = pd.DataFrame(fresh)
                st.session_state["fresh_overall"] = fresh_frame
                st.session_state["fresh_evaluations"] = evaluations
                st.session_state["fresh_dataset_name"] = selected_data_name
                st.success(f"Calculated metrics using {len(sample):,} rows from {selected_data_name}.")
        if "fresh_overall" in st.session_state:
            st.markdown(f"#### Fresh metrics — {st.session_state.get('fresh_dataset_name', 'selected dataset')}")
            display_table(st.session_state["fresh_overall"], "fresh_overall_metrics")
            st.pyplot(
                plot_metric_comparison(
                    st.session_state["fresh_overall"],
                    ["Accuracy (%)", "Balanced accuracy (%)", "Macro precision (%)",
                     "Macro recall (%)", "Macro F1 (%)"],
                ),
                use_container_width=True,
            )
            st.markdown("#### Per-class metrics")
            for result in st.session_state.get("fresh_evaluations", []):
                st.markdown(f"**{result['model']}**")
                display_table(result["per_class"], f"{result['model']}_per_class")

    st.markdown("### Metric definitions")
    definitions = pd.DataFrame([
        {"Metric": "Accuracy", "Definition": "Correct predictions divided by all predictions."},
        {"Metric": "Balanced accuracy", "Definition": "Average recall across classes; useful when class counts differ."},
        {"Metric": "Precision", "Definition": "Of predictions for a class, the fraction that are correct."},
        {"Metric": "Recall / sensitivity", "Definition": "Of true examples of a class, the fraction detected."},
        {"Metric": "Specificity", "Definition": "Of examples not belonging to a class, the fraction correctly rejected."},
        {"Metric": "F1-score", "Definition": "Harmonic mean of precision and recall."},
        {"Metric": "Macro average", "Definition": "Average across classes with equal weight per class."},
        {"Metric": "Weighted average", "Definition": "Average across classes weighted by class support."},
        {"Metric": "MCC", "Definition": "Matthews correlation coefficient; multiclass score ranges from -1 to +1."},
        {"Metric": "False-positive rate", "Definition": "False positives divided by all actual negatives for that class."},
        {"Metric": "False-negative rate", "Definition": "Missed positives divided by all actual positives for that class."},
    ])
    display_table(definitions, "metric_definitions")


# ---------------------------------------------------------------------------
# 8. Page: Confusion matrix
# ---------------------------------------------------------------------------

elif page == "Confusion Matrix":
    st.header("Confusion matrix and error analysis")
    st.write(
        "Rows represent actual classes and columns represent predicted classes. "
        "Off-diagonal cells are mistakes. A Normal row predicted as FDI/TSA is a "
        "false alarm; an attack row predicted as Normal is a missed attack."
    )

    mode = st.radio("Matrix source", ["Historical reference", "Fresh evaluation from a CSV"], horizontal=True)
    if mode == "Historical reference":
        selected_model = st.selectbox("Model", list(HISTORICAL_METRICS.keys()))
        values = HISTORICAL_METRICS[selected_model]
        classes = CLASS_ORDER
        matrix = np.asarray(values["matrix"])
        st.caption(values["description"])
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            image = ax.imshow(matrix, cmap="Blues")
            fig.colorbar(image, ax=ax)
            ax.set_xticks(range(len(classes)), labels=classes)
            ax.set_yticks(range(len(classes)), labels=classes)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Actual class")
            ax.set_title(f"{selected_model} — historical confusion matrix")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        with col2:
            row_total = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(matrix, row_total, out=np.zeros_like(matrix, dtype=float), where=row_total != 0) * 100
            fig, ax = plt.subplots(figsize=(6, 4.5))
            image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
            fig.colorbar(image, ax=ax)
            ax.set_xticks(range(len(classes)), labels=classes)
            ax.set_yticks(range(len(classes)), labels=classes)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Actual class")
            ax.set_title(f"{selected_model} — row-normalized (%)")
            for i in range(normalized.shape[0]):
                for j in range(normalized.shape[1]):
                    ax.text(j, i, f"{normalized[i, j]:.1f}%", ha="center", va="center")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        error_rows = []
        for i, actual in enumerate(classes):
            for j, predicted in enumerate(classes):
                error_rows.append({
                    "Actual": actual,
                    "Predicted": predicted,
                    "Count": int(matrix[i, j]),
                    "Meaning": "Correct" if i == j else f"{actual} classified as {predicted}",
                })
        display_table(pd.DataFrame(error_rows), "historical_confusion_cells")
    else:
        labelled_names = [name for name, frame in datasets.items() if LABEL_COLUMN in frame.columns]
        if not labelled_names or not models:
            st.warning("A labelled dataset and at least one loaded model are required.")
        else:
            selected_data_name = st.selectbox("Labelled dataset", labelled_names, key="cm_dataset")
            selected_model = st.selectbox("Model", list(models.keys()), key="cm_model")
            frame = datasets[selected_data_name]
            max_rows = min(50000, len(frame))
            count = st.slider("Rows to evaluate", 100, max(100, max_rows), min(15000, max_rows), step=100)
            if st.button("Evaluate and draw matrix"):
                sample = frame.sample(count, random_state=42) if len(frame) > count else frame
                try:
                    result = evaluate_frame(models[selected_model], sample, selected_model)
                    st.pyplot(plot_confusion_matrix(result), use_container_width=True)
                    st.pyplot(plot_confusion_matrix(result, normalize=True), use_container_width=True)
                    display_table(result["per_class"], "fresh_confusion_per_class")
                except Exception as exc:
                    st.error(str(exc))


# ---------------------------------------------------------------------------
# 9. Page: Attack simulator
# ---------------------------------------------------------------------------

elif page == "Attack Simulator":
    st.header("Attack simulator")
    st.write(
        "Enter one PMU feature row and compare predictions from the loaded models. "
        "This is a demonstration, not a replacement for a validated live monitoring system."
    )

    if not models:
        st.warning("Load at least one model before using the simulator.")
    else:
        defaults = {
            "Actual frequency value": 50.0,
            "Fraction of second": 0.5,
            "PMU time": 1000.0,
            "PDC time": 1000.01,
            "interarrival time": 0.01,
            "time difference": 0.01,
        }
        st.markdown("### Input PMU telemetry")
        inputs: Dict[str, float] = {}
        col1, col2 = st.columns(2)
        for index, feature in enumerate(FEATURES):
            target = col1 if index % 2 == 0 else col2
            with target:
                inputs[feature] = st.number_input(
                    feature,
                    value=float(defaults[feature]),
                    format="%.6f",
                    key=f"sim_{feature}",
                )

        preset = st.selectbox(
            "Optional demonstration preset",
            ["Manual values", "Normal-like sample", "Subtle frequency shift", "Time synchronization shift"],
        )
        if preset == "Normal-like sample":
            inputs["Actual frequency value"] = 50.0
            inputs["Fraction of second"] = 0.5
            inputs["PMU time"] = 1000.0
            inputs["PDC time"] = 1000.01
            inputs["interarrival time"] = 0.01
            inputs["time difference"] = 0.01
        elif preset == "Subtle frequency shift":
            inputs["Actual frequency value"] = 50.003
        elif preset == "Time synchronization shift":
            inputs["PMU time"] = 1300.0
            inputs["time difference"] = 300.0

        if st.button("Run prediction", type="primary"):
            one_row = pd.DataFrame([inputs])
            results = []
            for name, model in models.items():
                try:
                    start = time.perf_counter()
                    prediction = model_predict(model, one_row)[0]
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    results.append({
                        "Model": name,
                        "Predicted class": normalize_label(prediction),
                        "Prediction latency (ms)": round(elapsed_ms, 4),
                        "Features used": len(model_feature_names(model)),
                    })
                    probabilities, classes = model_probabilities(model, one_row)
                    if probabilities is not None and classes is not None:
                        probability_rows = []
                        for class_index, class_name in enumerate(classes):
                            probability_rows.append({
                                "Model": name,
                                "Class": class_name,
                                "predict_proba (%)": float(probabilities[0, class_index]) * 100,
                            })
                        st.markdown(f"**{name} — predict_proba output**")
                        display_table(pd.DataFrame(probability_rows), f"sim_probability_{name}")
                except Exception as exc:
                    results.append({"Model": name, "Error": str(exc)})
            display_table(pd.DataFrame(results), "simulator_predictions")
            st.warning(
                "The preset values are illustrative. A single manually entered row does not "
                "prove the model is robust, and predict_proba outputs are not necessarily calibrated."
            )


# ---------------------------------------------------------------------------
# 10. Page: Subtle attack analysis
# ---------------------------------------------------------------------------

elif page == "Subtle Attack Analysis":
    st.header("Subtle attack analysis")
    st.write(
        "Subtle attacks are evaluated separately because they can be harder to detect than "
        "the attacks represented in ordinary test data. Detection rate here means the share "
        "of rows labelled as an attack that the model predicts as that same attack class."
    )

    subtle_rows = []
    subtle_results = []
    subtle_dataset_labels = [
        ("Subtle FDI", "Subtle FDI test data", "FDI"),
        ("Subtle TSA", "Subtle TSA test data", "TSA"),
    ]
    for label, dataset_name, target_class in subtle_dataset_labels:
        frame = datasets.get(dataset_name)
        if frame is None:
            continue
        for model_name, model in models.items():
            try:
                predicted = model_predict(model, frame)
                normalized_pred = np.asarray([normalize_label(item) for item in predicted])
                detected = int(np.sum(normalized_pred == target_class))
                total = len(frame)
                detected_rate = safe_divide(detected, total) * 100
                predicted_counts = pd.Series(normalized_pred).value_counts()
                subtle_rows.append({
                    "Attack test": label,
                    "Model": model_name,
                    "Rows tested": total,
                    f"Predicted {target_class}": detected,
                    "Predicted Normal": int(predicted_counts.get("Normal", 0)),
                    "Predicted other attack": int(total - detected - predicted_counts.get("Normal", 0)),
                    "Detection rate (%)": detected_rate,
                    "Miss rate (%)": 100 - detected_rate,
                    "Source": "Fresh prediction on dataset file",
                })
                subtle_results.append({
                    "Attack test": label,
                    "Model": model_name,
                    "Detection rate (%)": detected_rate,
                    "Miss rate (%)": 100 - detected_rate,
                })
            except Exception as exc:
                st.error(f"{model_name} on {label}: {exc}")

    if subtle_rows:
        section_title("Table 5 — Fresh subtle-attack results")
        fresh_subtle = pd.DataFrame(subtle_rows)
        display_table(fresh_subtle, "fresh_subtle_attack_metrics")
        fig, ax = plt.subplots(figsize=(9, 4.8))
        pivot = fresh_subtle.pivot(index="Attack test", columns="Model", values="Detection rate (%)")
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Detection rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Subtle attack detection rate — freshly calculated")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    section_title("Historical subtle-attack reference")
    historical_rows = [
        {
            "Attack test": name,
            "Detected attacks": values["detected"],
            "Total samples": values["total"],
            "Historical detection rate (%)": values["detection_rate"],
            "Source": values["source"],
        }
        for name, values in HISTORICAL_SUBTLE.items()
    ]
    display_table(pd.DataFrame(historical_rows), "historical_subtle_metrics")

    st.info(
        "The historical subtle-FDI result was about 10.73% for the five-feature model, while "
        "the historical subtle-TSA result was about 99.94%. These are prior-run results, not "
        "guaranteed current outputs. The fresh table above uses the actual files and models found now."
    )

    st.markdown("### How the subtle test files were constructed")
    construction = pd.DataFrame([
        {"Test": "Subtle FDI", "Construction described in the project": "Small positive perturbation to frequency (approximately +0.001 to +0.005).", "Caution": "Synthetic distribution may differ from real attacks."},
        {"Test": "Subtle TSA", "Construction described in the project": "PMU timestamp shifted by 300 seconds modulo 3600; time difference recalculated.", "Caution": "A synthetic time shift does not cover every real synchronization attack."},
    ])
    display_table(construction, "subtle_construction")


# ---------------------------------------------------------------------------
# 11. Page: Upload and evaluate
# ---------------------------------------------------------------------------

elif page == "Upload & Evaluate":
    st.header("Upload a labelled CSV and evaluate both models")
    st.write(
        "Upload a CSV containing `attack_type` and the feature columns required by the "
        "selected model. This is the most direct way to calculate current metrics on your own test data."
    )

    uploaded = st.file_uploader("Upload labelled CSV", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_frame = pd.read_csv(uploaded)
            st.success(f"Loaded {len(uploaded_frame):,} rows and {uploaded_frame.shape[1]} columns.")
            st.markdown("### Table 6 — Uploaded dataset summary")
            display_table(pd.DataFrame([dataset_summary(uploaded_frame, uploaded.name)]), "uploaded_dataset_summary")
            st.dataframe(uploaded_frame.head(10), use_container_width=True, hide_index=True)

            if LABEL_COLUMN not in uploaded_frame.columns:
                st.error("The uploaded file must include an `attack_type` column for labelled evaluation.")
            elif not models:
                st.error("No model is currently loaded.")
            else:
                row_limit = min(len(uploaded_frame), 100000)
                count = st.slider(
                    "Rows to evaluate from upload",
                    min_value=100,
                    max_value=max(100, row_limit),
                    value=min(15000, max(100, row_limit)),
                    step=100,
                )
                if st.button("Evaluate uploaded CSV", type="primary"):
                    sample = uploaded_frame.sample(count, random_state=42) if len(uploaded_frame) > count else uploaded_frame
                    uploaded_overall = []
                    uploaded_class_rows = []
                    uploaded_evaluations = []
                    progress = st.progress(0)
                    for index, (model_name, model) in enumerate(models.items()):
                        try:
                            evaluation = evaluate_frame(model, sample, model_name)
                            uploaded_evaluations.append(evaluation)
                            uploaded_overall.append(build_overall_metrics(evaluation))
                            uploaded_class_rows.append(evaluation["per_class"])
                        except Exception as exc:
                            st.error(f"{model_name} failed: {exc}")
                        progress.progress((index + 1) / max(len(models), 1))
                    if uploaded_overall:
                        result_frame = pd.DataFrame(uploaded_overall)
                        st.session_state["uploaded_overall"] = result_frame
                        st.session_state["uploaded_per_class"] = pd.concat(uploaded_class_rows, ignore_index=True)
                        st.session_state["uploaded_evaluations"] = uploaded_evaluations
                        st.session_state["uploaded_name"] = uploaded.name
                        st.success("Evaluation completed.")
                if "uploaded_overall" in st.session_state:
                    st.markdown(f"### Overall metrics — {st.session_state.get('uploaded_name', 'upload')}")
                    display_table(st.session_state["uploaded_overall"], "uploaded_overall_metrics")
                    st.markdown("### Per-class precision, recall, F1, specificity, and error rates")
                    display_table(st.session_state["uploaded_per_class"], "uploaded_per_class_metrics")
                    st.pyplot(
                        plot_metric_comparison(
                            st.session_state["uploaded_overall"],
                            ["Accuracy (%)", "Balanced accuracy (%)", "Macro precision (%)",
                             "Macro recall (%)", "Macro F1 (%)"],
                        ),
                        use_container_width=True,
                    )
                    for evaluation in st.session_state.get("uploaded_evaluations", []):
                        st.markdown(f"#### {evaluation['model']}")
                        st.pyplot(plot_confusion_matrix(evaluation), use_container_width=True)
                        st.pyplot(plot_confusion_matrix(evaluation, normalize=True), use_container_width=True)
                        dataframe_download(
                            pd.DataFrame(evaluation["matrix"], index=evaluation["classes"], columns=evaluation["classes"]).reset_index().rename(columns={"index": "Actual"}),
                            f"{evaluation['model']}_confusion_matrix.csv".replace(" ", "_"),
                            "Download confusion matrix",
                        )
        except Exception as exc:
            st.error(f"Could not process uploaded CSV: {exc}")


# ---------------------------------------------------------------------------
# 12. Page: Feature analysis
# ---------------------------------------------------------------------------

elif page == "Feature Analysis":
    st.header("Feature analysis")
    if not datasets:
        st.error("No datasets were found.")
        st.stop()

    data_name = st.selectbox("Choose dataset", list(datasets.keys()), key="feature_dataset")
    frame = datasets[data_name]
    features_present = [column for column in FEATURES if column in frame.columns]
    if not features_present:
        st.warning("No known PMU feature columns were found in this dataset.")
        st.stop()

    selected_feature = st.selectbox("Feature", features_present)
    st.pyplot(plot_feature_histogram(frame, selected_feature), use_container_width=True)

    st.markdown("### Numeric summary")
    numeric = frame[features_present].apply(pd.to_numeric, errors="coerce")
    summary = numeric.describe().T.reset_index().rename(columns={"index": "Feature"})
    summary["Missing"] = numeric.isna().sum().values
    display_table(summary, "feature_numeric_summary")

    st.markdown("### Correlation matrix")
    st.pyplot(plot_correlation(frame), use_container_width=True)

    st.markdown("### Feature importance (if model supports it)")
    importance_rows = []
    for model_name, model in models.items():
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            continue
        names = model_feature_names(model)
        for feature_name, importance in zip(names, importances):
            importance_rows.append({
                "Model": model_name,
                "Feature": feature_name,
                "Importance": float(importance),
            })
    if importance_rows:
        importance_frame = pd.DataFrame(importance_rows)
        display_table(importance_frame.sort_values(["Model", "Importance"], ascending=[True, False]), "feature_importance")
        for model_name in importance_frame["Model"].unique():
            selected = importance_frame[importance_frame["Model"] == model_name].sort_values("Importance")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(selected["Feature"], selected["Importance"])
            ax.set_title(f"Feature importance — {model_name}")
            ax.set_xlabel("Importance")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
    else:
        st.info("The loaded models do not expose `feature_importances_`.")

    st.markdown("### Why feature selection matters")
    st.write(
        "The six-feature model uses `time difference`, while the five-feature experimental "
        "model omits it. This comparison can reveal whether the feature helps ordinary "
        "classification or causes the model to rely on a signal that changes under attack. "
        "Feature importance is not proof of causation and should be interpreted alongside test results."
    )


# ---------------------------------------------------------------------------
# 13. Page: Technical details
# ---------------------------------------------------------------------------

elif page == "Technical Details":
    st.header("Technical details")
    st.markdown("### Input feature schema")
    feature_info = pd.DataFrame([
        {"Feature": "Actual frequency value", "Description": "Measured frequency value recorded by the PMU.", "Potential concern": "Subtle shifts can be small relative to normal variation."},
        {"Feature": "Fraction of second", "Description": "Fractional timestamp component.", "Potential concern": "Distribution may depend on data collection and timestamp precision."},
        {"Feature": "PMU time", "Description": "Timestamp associated with the PMU measurement.", "Potential concern": "Synthetic or real time shifts can affect its distribution."},
        {"Feature": "PDC time", "Description": "Timestamp associated with the Phasor Data Concentrator.", "Potential concern": "Clock alignment and network delay can affect values."},
        {"Feature": "interarrival time", "Description": "Time interval between successive arrivals.", "Potential concern": "Network congestion and sampling can change timing."},
        {"Feature": "time difference", "Description": "Time difference feature used by the six-feature model.", "Potential concern": "May leak an attack-related shortcut or shift between test distributions."},
    ])
    display_table(feature_info, "feature_schema")

    st.markdown("### Model inventory")
    inventory_rows = []
    for name, path in MODEL_FILES.items():
        model = models.get(name)
        inventory_rows.append({
            "Model": name,
            "File": str(path.relative_to(ROOT)),
            "Available": model is not None,
            "Features": ", ".join(model_feature_names(model)) if model is not None else "N/A",
            "Feature count": getattr(model, "n_features_in_", "N/A") if model is not None else "N/A",
            "Supports predict_proba": bool(hasattr(model, "predict_proba")) if model is not None else False,
            "Supports feature importance": bool(hasattr(model, "feature_importances_")) if model is not None else False,
        })
    display_table(pd.DataFrame(inventory_rows), "model_inventory")

    st.markdown("### Project workflow")
    workflow = pd.DataFrame([
        {"Step": 1, "Stage": "PMU data", "Output": "CSV telemetry with labels and timing/frequency features."},
        {"Step": 2, "Stage": "Preprocessing", "Output": "Clean numeric feature columns and attack labels."},
        {"Step": 3, "Stage": "Model training", "Output": "Saved Random Forest model files in models/."},
        {"Step": 4, "Stage": "Prediction", "Output": "Normal, FDI, or TSA class prediction."},
        {"Step": 5, "Stage": "Evaluation", "Output": "Accuracy, class metrics, confusion matrix, and false alarms/misses."},
        {"Step": 6, "Stage": "Subtle attack test", "Output": "Separate detection rates for synthetic subtle FDI and TSA files."},
    ])
    display_table(workflow, "project_workflow")

    st.markdown("### Reliability and limitations")
    limitations = pd.DataFrame([
        {"Limitation": "Synthetic subtle attacks", "Why it matters": "Synthetic perturbations do not represent every real attack distribution."},
        {"Limitation": "Dataset shift", "Why it matters": "Scores can change when the source, operating conditions, or class proportions change."},
        {"Limitation": "Feature leakage", "Why it matters": "Timing-derived features may encode collection or labelling shortcuts."},
        {"Limitation": "Probability calibration", "Why it matters": "Random Forest predict_proba values are not automatically calibrated probabilities."},
        {"Limitation": "Evaluation data", "Why it matters": "Metrics are only meaningful when labels are correct and test data is independent of training data."},
        {"Limitation": "Operational deployment", "Why it matters": "This dashboard is a project demonstrator, not a certified protective relay or live production IDS."},
    ])
    display_table(limitations, "limitations")

    st.markdown("### Environment")
    env_rows = [
        {"Item": "Python", "Value": __import__("sys").version.split()[0]},
        {"Item": "Streamlit", "Value": st.__version__},
        {"Item": "Root directory", "Value": str(ROOT)},
        {"Item": "Dataset directory exists", "Value": str(DATASET_DIR.exists())},
        {"Item": "Model directory exists", "Value": str(MODEL_DIR.exists())},
    ]
    display_table(pd.DataFrame(env_rows), "environment_info")

    st.markdown("### Export report data")
    export_rows = []
    for name, values in HISTORICAL_METRICS.items():
        for metric, value in values.items():
            if metric == "matrix":
                continue
            export_rows.append({"Model": name, "Metric": metric, "Value": value})
    historical_export = pd.DataFrame(export_rows)
    dataframe_download(historical_export, "smartgrid_historical_metrics.csv", "Download historical metrics")


# ---------------------------------------------------------------------------
# 14. Footer and persistent metric reminders
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "SmartGrid-AIDM academic demonstrator • Metrics shown as historical values are labelled "
    "as such; freshly computed metrics depend on the current models and selected dataset."
)
st.caption(
    "For a fair comparison, evaluate both models on the same independent labelled test rows. "
    "Do not compare scores obtained from different datasets or confuse overall accuracy with attack detection rate."
)
