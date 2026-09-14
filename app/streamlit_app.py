import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Time-Series Anomaly Detection",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "nab"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"


# =========================================================
# FROZEN ANOMALY DETECTION THRESHOLDS
# =========================================================

ARIMA_THRESHOLD = 0.07615334829925355

ISOLATION_FOREST_THRESHOLD = 0.08179236078011909

LSTM_THRESHOLD = 0.020251801330746014

HYBRID_THRESHOLD = 0.9321572769953052


# =========================================================
# LOAD RAW TIME-SERIES DATA
# =========================================================

RAW_DATA_PATH = (
    DATA_PATH /
    "ec2_cpu_utilization_24ae8d.csv"
)


@st.cache_data
def load_raw_data():

    raw_df = pd.read_csv(
        RAW_DATA_PATH
    )

    raw_df["timestamp"] = pd.to_datetime(
        raw_df["timestamp"]
    )

    raw_df = (
        raw_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return raw_df


raw_df = load_raw_data()


# =========================================================
# LOAD MODEL RESULTS
# =========================================================

@st.cache_data
def load_data():

    arima = pd.read_csv(
        PROCESSED_PATH / "arima_test_results.csv"
    )

    isolation_forest = pd.read_csv(
        PROCESSED_PATH / "isolation_forest_test_results.csv"
    )

    lstm = pd.read_csv(
        PROCESSED_PATH / "lstm_test_results.csv"
    )

    hybrid = pd.read_csv(
        PROCESSED_PATH / "hybrid_test_results.csv"
    )

    # Convert timestamps
    arima["timestamp"] = pd.to_datetime(
        arima["timestamp"]
    )

    isolation_forest["timestamp"] = pd.to_datetime(
        isolation_forest["timestamp"]
    )

    lstm["timestamp"] = pd.to_datetime(
        lstm["timestamp"]
    )

    hybrid["timestamp"] = pd.to_datetime(
        hybrid["timestamp"]
    )

    return (
        arima,
        isolation_forest,
        lstm,
        hybrid
    )


(
    arima,
    isolation_forest,
    lstm,
    hybrid
) = load_data()


# =========================================================
# LOAD OFFICIAL NAB ANOMALY WINDOWS
# =========================================================

WINDOWS_PATH = (
    DATA_PATH /
    "combined_windows.json"
)


with open(
    WINDOWS_PATH,
    "r"
) as f:

    nab_windows = json.load(f)


DATASET_NAME = (
    "realAWSCloudwatch/"
    "ec2_cpu_utilization_24ae8d.csv"
)


anomaly_windows = [
    (
        pd.to_datetime(start),
        pd.to_datetime(end)
    )

    for start, end
    in nab_windows[DATASET_NAME]
]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🔎 Anomaly Analysis"
)


selected_model = st.sidebar.selectbox(
    "Select detection model",
    [
        "All Models",
        "ARIMA Residual",
        "Isolation Forest V2",
        "LSTM Autoencoder",
        "Hybrid V2"
    ]
)


selected_period = st.sidebar.selectbox(
    "Select analysis period",
    [
        "Full Test Period",
        "Anomaly Event 1",
        "Anomaly Event 2"
    ]
)


st.sidebar.divider()


st.sidebar.markdown(
    """
    **Project**

    Intelligent Time-Series Forecasting
    and Anomaly Detection

    **Dataset**

    Numenta Anomaly Benchmark (NAB)

    **Series**

    EC2 CPU Utilization
    """
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "📊 Intelligent Time-Series Forecasting & Anomaly Detection"
)


st.markdown(
    """
    ### Interactive Analysis Dashboard

    This dashboard presents the results of statistical,
    machine learning, deep learning, and hybrid approaches
    for time-series forecasting and anomaly detection.

    **Dataset:** Numenta Anomaly Benchmark (NAB) —
    EC2 CPU Utilization
    """
)


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.header(
    "Dataset Overview"
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Test Observations",
    f"{len(arima):,}"
)


col2.metric(
    "Sampling Interval",
    "5 minutes"
)


col3.metric(
    "Test Period",
    f"{arima['timestamp'].min().date()} → "
    f"{arima['timestamp'].max().date()}"
)


col4.metric(
    "Models Evaluated",
    "6"
)


# =========================================================
# FORECASTING ANALYSIS
# =========================================================

st.header(
    "📊 Forecasting Analysis"
)


st.markdown(
    """
    Forecasting performance is compared using a simple
    Naive baseline and the selected ARIMA model.

    The final test period is evaluated using Mean Absolute
    Error (MAE) and Root Mean Squared Error (RMSE).
    """
)


# ---------------------------------------------------------
# Forecasting metrics
# ---------------------------------------------------------

naive_mae = 0.0509789343

naive_rmse = 0.1751726923

arima_mae = 0.031119

arima_rmse = 0.123077


mae_improvement = (
    (naive_mae - arima_mae)
    / naive_mae
) * 100


rmse_improvement = (
    (naive_rmse - arima_rmse)
    / naive_rmse
) * 100


# ---------------------------------------------------------
# Metric cards
# ---------------------------------------------------------

fc1, fc2, fc3, fc4 = st.columns(4)


fc1.metric(
    "Naive MAE",
    f"{naive_mae:.4f}"
)


fc2.metric(
    "ARIMA MAE",
    f"{arima_mae:.4f}",
    f"{mae_improvement:.1f}% improvement"
)


fc3.metric(
    "Naive RMSE",
    f"{naive_rmse:.4f}"
)


fc4.metric(
    "ARIMA RMSE",
    f"{arima_rmse:.4f}",
    f"{rmse_improvement:.1f}% improvement"
)


# ---------------------------------------------------------
# Forecast comparison table
# ---------------------------------------------------------

forecast_results = pd.DataFrame({

    "Model": [
        "Naive Forecast",
        "ARIMA (1,0,0)"
    ],

    "MAE": [
        naive_mae,
        arima_mae
    ],

    "RMSE": [
        naive_rmse,
        arima_rmse
    ]
})


st.dataframe(
    forecast_results.style.format({

        "MAE": "{:.4f}",

        "RMSE": "{:.4f}"

    }),

    use_container_width=True,

    hide_index=True
)


# =========================================================
# FORECAST PERIOD SELECTION
# =========================================================

if selected_period == "Anomaly Event 1":

    forecast_start = (
        anomaly_windows[0][0]
        - pd.Timedelta(hours=6)
    )

    forecast_end = (
        anomaly_windows[0][1]
        + pd.Timedelta(hours=6)
    )


elif selected_period == "Anomaly Event 2":

    forecast_start = (
        anomaly_windows[1][0]
        - pd.Timedelta(hours=6)
    )

    forecast_end = (
        anomaly_windows[1][1]
        + pd.Timedelta(hours=6)
    )


else:

    forecast_start = (
        arima["timestamp"].min()
    )

    forecast_end = (
        arima["timestamp"].max()
    )


# =========================================================
# PREPARE FORECAST DATA
# =========================================================

forecast_plot = arima[
    (arima["timestamp"] >= forecast_start)
    &
    (arima["timestamp"] <= forecast_end)
].copy()


# =========================================================
# NAIVE FORECAST
# =========================================================

forecast_plot["naive_forecast"] = (
    forecast_plot["value"].shift(1)
)


# =========================================================
# FIRST NAIVE FORECAST VALUE
# =========================================================

if not forecast_plot.empty:

    first_timestamp = (
        forecast_plot["timestamp"].iloc[0]
    )

    previous_timestamp = (
        first_timestamp
        - pd.Timedelta(minutes=5)
    )

    previous_value = raw_df.loc[
        raw_df["timestamp"] == previous_timestamp,
        "value"
    ]

    if not previous_value.empty:

        forecast_plot.loc[
            forecast_plot.index[0],
            "naive_forecast"
        ] = previous_value.iloc[0]


# =========================================================
# FORECAST VISUALIZATION
# =========================================================

st.subheader(
    "Forecast vs Actual — Selected Period"
)


fig_forecast, ax_forecast = plt.subplots(
    figsize=(15, 6)
)


# ---------------------------------------------------------
# Actual
# ---------------------------------------------------------

ax_forecast.plot(
    forecast_plot["timestamp"],
    forecast_plot["value"],
    linewidth=1,
    label="Actual"
)


# ---------------------------------------------------------
# ARIMA
# ---------------------------------------------------------

ax_forecast.plot(
    forecast_plot["timestamp"],
    forecast_plot["forecast"],
    linewidth=1.5,
    label="ARIMA Forecast"
)


# ---------------------------------------------------------
# Naive
# ---------------------------------------------------------

ax_forecast.plot(
    forecast_plot["timestamp"],
    forecast_plot["naive_forecast"],
    linewidth=1,
    alpha=0.8,
    label="Naive Forecast"
)


# =========================================================
# SHADE OFFICIAL ANOMALY WINDOWS
# =========================================================

for i, (start, end) in enumerate(
    anomaly_windows
):

    clipped_start = max(
        start,
        forecast_start
    )

    clipped_end = min(
        end,
        forecast_end
    )

    if clipped_start <= clipped_end:

        ax_forecast.axvspan(
            clipped_start,
            clipped_end,
            alpha=0.20,
            label=(
                "Official NAB Anomaly Window"
                if i == 0
                else None
            )
        )


# =========================================================
# FORMATTING
# =========================================================

ax_forecast.set_title(
    "Actual vs Naive and ARIMA Forecasts"
)


ax_forecast.set_xlabel(
    "Time"
)


ax_forecast.set_ylabel(
    "CPU Utilization"
)


ax_forecast.grid(
    alpha=0.3
)


ax_forecast.legend(
    loc="upper left"
)


plt.xticks(
    rotation=30
)


plt.tight_layout()


st.pyplot(
    fig_forecast,
    use_container_width=True
)


# =========================================================
# ANOMALY DETECTION GRAPH
# =========================================================

st.header(
    "🚨 Anomaly Detection — Model Comparison"
)


st.markdown(
    """
    The graph compares actual EC2 CPU utilization with the
    official NAB anomaly windows and the anomalies detected
    by the selected model.
    """
)


# =========================================================
# SELECT TIME PERIOD
# =========================================================

if selected_period == "Anomaly Event 1":

    plot_start = (
        anomaly_windows[0][0]
        - pd.Timedelta(hours=6)
    )

    plot_end = (
        anomaly_windows[0][1]
        + pd.Timedelta(hours=6)
    )


elif selected_period == "Anomaly Event 2":

    plot_start = (
        anomaly_windows[1][0]
        - pd.Timedelta(hours=6)
    )

    plot_end = (
        anomaly_windows[1][1]
        + pd.Timedelta(hours=6)
    )


else:

    plot_start = (
        arima["timestamp"].min()
    )

    plot_end = (
        arima["timestamp"].max()
    )


plot_mask = (
    (arima["timestamp"] >= plot_start)
    &
    (arima["timestamp"] <= plot_end)
)


plot_arima = arima[
    plot_mask
].copy()


fig, ax = plt.subplots(
    figsize=(15, 6)
)


# =========================================================
# ACTUAL CPU UTILIZATION
# =========================================================

ax.plot(
    plot_arima["timestamp"],
    plot_arima["value"],
    linewidth=1,
    label="Actual CPU Utilization"
)


# =========================================================
# OFFICIAL NAB ANOMALY WINDOWS
# =========================================================

visible_windows = []


for start, end in anomaly_windows:

    clipped_start = max(
        start,
        plot_start
    )

    clipped_end = min(
        end,
        plot_end
    )

    if clipped_start <= clipped_end:

        visible_windows.append(
            (
                clipped_start,
                clipped_end
            )
        )


for i, (start, end) in enumerate(
    visible_windows
):

    ax.axvspan(
        start,
        end,
        alpha=0.20,
        label=(
            "Official NAB Anomaly Window"
            if i == 0
            else None
        )
    )


# =========================================================
# FILTER FUNCTION
# =========================================================

def filter_period(data):

    return data[
        (data["timestamp"] >= plot_start)
        &
        (data["timestamp"] <= plot_end)
    ].copy()


# =========================================================
# ARIMA DETECTIONS
# =========================================================

if selected_model in [
    "All Models",
    "ARIMA Residual"
]:

    arima_anomalies = arima[
        arima["absolute_error"]
        >= ARIMA_THRESHOLD
    ]

    arima_anomalies = filter_period(
        arima_anomalies
    )

    ax.scatter(
        arima_anomalies["timestamp"],
        arima_anomalies["value"],
        marker="x",
        s=55,
        label="ARIMA Detection"
    )


# =========================================================
# ISOLATION FOREST V2 DETECTIONS
# =========================================================

if selected_model in [
    "All Models",
    "Isolation Forest V2"
]:

    if_anomalies = isolation_forest[
        isolation_forest["predicted_anomaly"]
        == 1
    ]

    if_anomalies = filter_period(
        if_anomalies
    )

    if_values = arima[
        arima["timestamp"].isin(
            if_anomalies["timestamp"]
        )
    ]

    ax.scatter(
        if_values["timestamp"],
        if_values["value"],
        marker="^",
        s=55,
        label="Isolation Forest V2"
    )


# =========================================================
# LSTM AUTOENCODER DETECTIONS
# =========================================================

if selected_model in [
    "All Models",
    "LSTM Autoencoder"
]:

    lstm_anomalies = lstm[
        lstm["predicted_anomaly"]
        == 1
    ]

    lstm_anomalies = filter_period(
        lstm_anomalies
    )

    lstm_values = arima[
        arima["timestamp"].isin(
            lstm_anomalies["timestamp"]
        )
    ]

    ax.scatter(
        lstm_values["timestamp"],
        lstm_values["value"],
        marker="s",
        s=55,
        label="LSTM Autoencoder"
    )


# =========================================================
# HYBRID V2 DETECTIONS
# =========================================================

if selected_model in [
    "All Models",
    "Hybrid V2"
]:

    hybrid_anomalies = hybrid[
        hybrid["hybrid_v2_score"]
        >= HYBRID_THRESHOLD
    ]

    hybrid_anomalies = filter_period(
        hybrid_anomalies
    )

    ax.scatter(
        hybrid_anomalies["timestamp"],
        hybrid_anomalies["value"],
        marker="*",
        s=90,
        label="Hybrid V2"
    )


# =========================================================
# GRAPH FORMATTING
# =========================================================

ax.set_xlabel(
    "Time"
)


ax.set_ylabel(
    "CPU Utilization"
)


ax.set_title(
    "EC2 CPU Utilization with Official "
    "Anomaly Windows and Model Detections"
)


ax.grid(
    alpha=0.3
)


ax.legend(
    loc="upper left",
    fontsize=9
)


plt.xticks(
    rotation=30
)


plt.tight_layout()


st.pyplot(
    fig,
    use_container_width=True
)


# =========================================================
# DETECTION SUMMARY
# =========================================================

st.header(
    "🚨 Detection Summary"
)


summary_data = []


# ---------------------------------------------------------
# ARIMA
# ---------------------------------------------------------

arima_count = int(
    (
        arima["absolute_error"]
        >= ARIMA_THRESHOLD
    ).sum()
)


summary_data.append(
    [
        "ARIMA Residual",
        arima_count
    ]
)


# ---------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------

if_count = int(
    (
        isolation_forest["predicted_anomaly"]
        == 1
    ).sum()
)


summary_data.append(
    [
        "Isolation Forest V2",
        if_count
    ]
)


# ---------------------------------------------------------
# LSTM
# ---------------------------------------------------------

lstm_count = int(
    (
        lstm["predicted_anomaly"]
        == 1
    ).sum()
)


summary_data.append(
    [
        "LSTM Autoencoder",
        lstm_count
    ]
)


# ---------------------------------------------------------
# Hybrid
# ---------------------------------------------------------

hybrid_count = int(
    (
        hybrid["hybrid_v2_score"]
        >= HYBRID_THRESHOLD
    ).sum()
)


summary_data.append(
    [
        "Hybrid V2",
        hybrid_count
    ]
)


summary_df = pd.DataFrame(
    summary_data,
    columns=[
        "Model",
        "Detected Anomalies"
    ]
)


st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.header(
    "📈 Model Performance"
)


st.markdown(
    """
    Models are evaluated at two levels:

    **Point-level:** performance on individual observations.

    **Window-level:** performance relative to the official
    NAB anomaly windows.
    """
)


model_results = pd.DataFrame({

    "Model": [
        "ARIMA Residual",
        "Isolation Forest V1",
        "Isolation Forest V2",
        "LSTM Autoencoder",
        "Hybrid V1",
        "Hybrid V2"
    ],

    "Point Precision": [
        0.0870,
        0.0526,
        0.0556,
        0.0286,
        0.0606,
        0.0500
    ],

    "Point Recall": [
        1.0000,
        0.5000,
        1.0000,
        0.5000,
        1.0000,
        1.0000
    ],

    "Point F1": [
        0.1600,
        0.0952,
        0.1053,
        0.0541,
        0.1143,
        0.0952
    ],

    "Point ROC-AUC": [
        0.9981,
        0.9845,
        0.9795,
        0.9298,
        0.9904,
        0.9950
    ],

    "Window Precision": [
        0.4783,
        0.6842,
        0.7222,
        0.8286,
        0.6667,
        0.6250
    ],

    "Window Recall": [
        0.0274,
        0.0323,
        0.0647,
        0.0721,
        0.0547,
        0.0622
    ],

    "Window F1": [
        0.0518,
        0.0618,
        0.1187,
        0.1327,
        0.1011,
        0.1131
    ],

    "Event Detection Rate": [
        1.00,
        1.00,
        1.00,
        0.50,
        1.00,
        1.00
    ]
})


st.dataframe(
    model_results.style.format({

        "Point Precision": "{:.4f}",

        "Point Recall": "{:.4f}",

        "Point F1": "{:.4f}",

        "Point ROC-AUC": "{:.4f}",

        "Window Precision": "{:.4f}",

        "Window Recall": "{:.4f}",

        "Window F1": "{:.4f}",

        "Event Detection Rate": "{:.0%}"

    }),

    use_container_width=True,

    hide_index=True
)


# =========================================================
# KEY RESULTS
# =========================================================

st.subheader(
    "🏆 Key Results"
)


best_window_f1 = model_results.loc[
    model_results["Window F1"].idxmax()
]


best_point_f1 = model_results.loc[
    model_results["Point F1"].idxmax()
]


best_auc = model_results.loc[
    model_results["Point ROC-AUC"].idxmax()
]


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Best Window F1",
    f"{best_window_f1['Window F1']:.4f}",
    best_window_f1["Model"]
)


col2.metric(
    "Best Point F1",
    f"{best_point_f1['Point F1']:.4f}",
    best_point_f1["Model"]
)


col3.metric(
    "Best ROC-AUC",
    f"{best_auc['Point ROC-AUC']:.4f}",
    best_auc["Model"]
)


col4.metric(
    "100% Event Detection",
    "5 / 6 Models",
    "Both anomaly events"
)


# =========================================================
# TEST DATA PREVIEW
# =========================================================

st.header(
    "🔍 Test Data Preview"
)


with st.expander(
    "View ARIMA test results"
):

    st.dataframe(
        arima[
            [
                "timestamp",
                "value",
                "forecast",
                "absolute_error",
                "ground_truth"
            ]
        ],

        use_container_width=True,

        hide_index=True
    )


# =========================================================
# MODEL AGREEMENT & EXPLAINABILITY
# =========================================================

st.header(
    "🧠 Model Agreement & Explainability"
)


st.markdown(
    """
    This section explains why individual observations are flagged
    as anomalies. Each detection model provides an independent
    signal, allowing us to examine the agreement between models.
    """
)


# =========================================================
# CREATE MODEL AGREEMENT DATA
# =========================================================

agreement_df = pd.DataFrame({

    "timestamp": arima["timestamp"],

    "value": arima["value"],

    "ARIMA": (
        arima["absolute_error"]
        >= ARIMA_THRESHOLD
    ).astype(int),

    "Isolation Forest V2": (
        isolation_forest["predicted_anomaly"]
        == 1
    ).astype(int),

    "LSTM Autoencoder": (
        lstm["predicted_anomaly"]
        == 1
    ).astype(int),

    "Hybrid V2": (
        hybrid["hybrid_v2_score"]
        >= HYBRID_THRESHOLD
    ).astype(int)
})


# =========================================================
# MODEL AGREEMENT COUNT
# =========================================================

agreement_df["Model Agreement"] = (

    agreement_df["ARIMA"]

    + agreement_df["Isolation Forest V2"]

    + agreement_df["LSTM Autoencoder"]
)


# =========================================================
# FILTER TO SELECTED PERIOD
# =========================================================

agreement_display = agreement_df[
    (agreement_df["timestamp"] >= plot_start)
    &
    (agreement_df["timestamp"] <= plot_end)
].copy()


# =========================================================
# SUMMARY METRICS
# =========================================================

total_points = len(
    agreement_display
)


points_with_agreement = int(
    (
        agreement_display["Model Agreement"]
        >= 2
    ).sum()
)


points_with_any_detection = int(
    (
        agreement_display["Model Agreement"]
        >= 1
    ).sum()
)


max_agreement = int(
    agreement_display["Model Agreement"].max()
) if total_points > 0 else 0


col1, col2, col3 = st.columns(3)


col1.metric(
    "Points with Any Detection",
    points_with_any_detection
)


col2.metric(
    "Points with ≥2 Model Agreement",
    points_with_agreement
)


col3.metric(
    "Maximum Model Agreement",
    f"{max_agreement} / 3"
)


# =========================================================
# AGREEMENT DISTRIBUTION
# =========================================================

st.subheader(
    "Model Agreement Distribution"
)


agreement_counts = (
    agreement_display["Model Agreement"]
    .value_counts()
    .sort_index()
)


agreement_table = pd.DataFrame({

    "Models Flagging":
        agreement_counts.index,

    "Number of Observations":
        agreement_counts.values
})


agreement_table["Models Flagging"] = (

    agreement_table["Models Flagging"]
    .astype(str)
    + " / 3"
)


st.dataframe(
    agreement_table,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# DETAILED DETECTION TABLE
# =========================================================

st.subheader(
    "🔍 Detection Explanation"
)


explanation_df = agreement_display[
    agreement_display["Model Agreement"] >= 1
].copy()


# ---------------------------------------------------------
# Convert binary values to readable labels
# ---------------------------------------------------------

for column in [
    "ARIMA",
    "Isolation Forest V2",
    "LSTM Autoencoder",
    "Hybrid V2"
]:

    explanation_df[column] = (
        explanation_df[column]
        .map({
            0: "Normal",
            1: "🚨 Anomaly"
        })
    )


explanation_df = explanation_df[
    [
        "timestamp",
        "value",
        "ARIMA",
        "Isolation Forest V2",
        "LSTM Autoencoder",
        "Hybrid V2",
        "Model Agreement"
    ]
].sort_values(
    "Model Agreement",
    ascending=False
)


with st.expander(
    "View detailed detection explanations"
):

    st.dataframe(
        explanation_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# ANOMALY SCORE ANALYSIS
# =========================================================

st.header(
    "📉 Anomaly Score Analysis"
)


st.markdown(
    """
    Anomaly scores indicate how unusual an observation appears
    to each detection model.

    Scores are normalized by their respective detection
    thresholds.

    **Interpretation:**

    • Score < 1 → below the anomaly threshold

    • Score = 1 → exactly at the anomaly threshold

    • Score > 1 → above the anomaly threshold

    Higher scores indicate stronger anomaly evidence.
    """
)


# =========================================================
# PREPARE NORMALIZED SCORES
# =========================================================

score_df = pd.DataFrame({

    "timestamp": arima["timestamp"],

    "ARIMA": (
        arima["absolute_error"]
        / ARIMA_THRESHOLD
    ),

    "Isolation Forest V2": (
        isolation_forest["anomaly_score"]
        / ISOLATION_FOREST_THRESHOLD
    ),

    "LSTM Autoencoder": (
        lstm["reconstruction_error"]
        / LSTM_THRESHOLD
    ),

    "Hybrid V2": (
        hybrid["hybrid_v2_score"]
        / HYBRID_THRESHOLD
    )
})


# =========================================================
# FILTER SELECTED PERIOD
# =========================================================

score_display = score_df[
    (score_df["timestamp"] >= plot_start)
    &
    (score_df["timestamp"] <= plot_end)
].copy()


# =========================================================
# CREATE SCORE PLOT
# =========================================================

fig_score, ax_score = plt.subplots(
    figsize=(15, 6)
)


# =========================================================
# PLOT SELECTED MODEL(S)
# =========================================================

if selected_model == "All Models":

    ax_score.plot(
        score_display["timestamp"],
        score_display["ARIMA"],
        linewidth=1,
        label="ARIMA"
    )

    ax_score.plot(
        score_display["timestamp"],
        score_display["Isolation Forest V2"],
        linewidth=1,
        label="Isolation Forest V2"
    )

    ax_score.plot(
        score_display["timestamp"],
        score_display["LSTM Autoencoder"],
        linewidth=1,
        label="LSTM Autoencoder"
    )

    ax_score.plot(
        score_display["timestamp"],
        score_display["Hybrid V2"],
        linewidth=1.5,
        label="Hybrid V2"
    )


else:

    ax_score.plot(
        score_display["timestamp"],
        score_display[selected_model],
        linewidth=1.5,
        label=selected_model
    )


# =========================================================
# DETECTION THRESHOLD
# =========================================================

ax_score.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
    label="Detection Threshold"
)


# =========================================================
# FORMAT SCORE GRAPH
# =========================================================

ax_score.set_title(
    "Normalized Anomaly Scores"
)


ax_score.set_xlabel(
    "Time"
)


ax_score.set_ylabel(
    "Score / Detection Threshold"
)


ax_score.grid(
    alpha=0.3
)


ax_score.legend(
    loc="upper left",
    fontsize=9
)


plt.xticks(
    rotation=30
)


plt.tight_layout()


st.pyplot(
    fig_score,
    use_container_width=True
)


# =========================================================
# SCORE STATISTICS
# =========================================================

st.subheader(
    "📊 Score Statistics"
)


score_summary = pd.DataFrame({

    "Model": [
        "ARIMA",
        "Isolation Forest V2",
        "LSTM Autoencoder",
        "Hybrid V2"
    ],

    "Maximum Score": [
        score_display["ARIMA"].max(),
        score_display["Isolation Forest V2"].max(),
        score_display["LSTM Autoencoder"].max(),
        score_display["Hybrid V2"].max()
    ],

    "Mean Score": [
        score_display["ARIMA"].mean(),
        score_display["Isolation Forest V2"].mean(),
        score_display["LSTM Autoencoder"].mean(),
        score_display["Hybrid V2"].mean()
    ],

    "Points Above Threshold": [

        int(
            (
                score_display["ARIMA"]
                >= 1
            ).sum()
        ),

        int(
            (
                score_display["Isolation Forest V2"]
                >= 1
            ).sum()
        ),

        int(
            (
                score_display["LSTM Autoencoder"]
                >= 1
            ).sum()
        ),

        int(
            (
                score_display["Hybrid V2"]
                >= 1
            ).sum()
        )
    ]
})


st.dataframe(
    score_summary.style.format({

        "Maximum Score": "{:.2f}",

        "Mean Score": "{:.2f}"

    }),

    use_container_width=True,

    hide_index=True
)


# =========================================================
# SCORE INTERPRETATION
# =========================================================

st.info(
    """
    **Interpretation:** The dashed line represents the frozen
    detection threshold. A point above this line has anomaly
    evidence stronger than the model's detection threshold.
    """
)


# =========================================================
# FOOTER
# =========================================================

st.divider()


st.caption(
    "Intelligent Time-Series Forecasting and Anomaly Detection "
    "| Numenta Anomaly Benchmark | IIT Guwahati"
)