# 📈 Predicciones y Resultados

Esta carpeta contiene las predicciones generadas por el pipeline multimodal.

## 📁 Estructura

```
predicciones/
├── submission_YYYY-MM-DD-HHMM.csv       # Predicciones para test
├── validation_predictions.csv           # Predicciones para validación
└── analysis/                           # Análisis detallado
    ├── confusion_matrix.png
    └── error_analysis.csv
```

## 📊 Formato de Predicciones

### Archivo de Submission (Test)
```csv
observationId,predictions
obs_001,"1234 5678 9012 3456 7890"
obs_002,"2345 6789 0123 4567 8901"
```

- **observationId**: ID de la observación de test
- **predictions**: Top-10 especies predichas (separadas por espacios)

## 🚀 Generación de Predicciones

### Desde la Aplicación
```bash
python app/main.py
# Usar interfaz web para análisis individual
```

### Desde el Pipeline
```python
from app.pipeline import MultimodalFungiCLEF2025Pipeline

pipeline = MultimodalFungiCLEF2025Pipeline(".")
train_data, val_data, index_data = pipeline.prepare_data_and_index()

# Generar predicciones para test
results_df = pipeline.predict_test(
    train_data, 
    index_data, 
    output_file='predicciones/submission_latest.csv',
    k=10
)
```
