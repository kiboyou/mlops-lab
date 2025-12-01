# validate_model.py
import pandas as pd
import joblib
import os
from evidently.report import Report
from evidently.metric_preset import ClassificationPerformancePreset

os.makedirs("reports", exist_ok=True)
# ... (Chargement des données et du modèle) ...

current_data['prediction'] = model.predict(current_data[feature_names])

model_performance_report = Report(metrics=[
    ClassificationPerformancePreset(target_names=list(model.classes_.astype(str))),
])

model_performance_report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=None
)
model_performance_report.save_html("reports/model_performance_report.html")