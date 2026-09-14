TimeSeries_Anomaly_Project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── nab/
│   │   ├── ec2_cpu_utilization_24ae8d.csv
│   │   ├── combined_labels.json
│   │   └── combined_windows.json
│   │
│   └── processed/
│       ├── arima_test_results.csv
│       ├── isolation_forest_test_results.csv
│       ├── lstm_test_results.csv
│       └── hybrid_test_results.csv
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── app/
│   └── streamlit_app.py
│
├── figures/
│   ├── 01_time_series_overview.png
│   ├── 02_cpu_distribution.png
│   ├── 03_rolling_statistics.png
│   ├── 04_acf.png
│   ├── 05_pacf.png
│   ├── 06_naive_vs_arima_forecast.png
│   ├── 07_arima_anomaly_detection.png
│   ├── 08_isolation_forest_v2.png
│   ├── 09_lstm_autoencoder.png
│   ├── 10_hybrid_v2.png
│   ├── 11_model_performance_comparison.png
│   ├── 12_model_agreement.png
│   ├── 13_anomaly_score_analysis.png
│   ├── 14_anomaly_events.png
│   └── streamlit_dashboard.png
│
└── report/
    ├── project_report.md
    └── project_report_AUDITED.docx