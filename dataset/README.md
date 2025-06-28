# 📊 Dataset FungiCLEF 2025

## 📋 Descripción General

Este proyecto utiliza el dataset **FungiCLEF 2025 FungiTastic-FewShot** proporcionado por LifeCLEF.

### 📈 Estadísticas del Dataset
- **Total de especies**: 2,427 especies únicas
- **Imágenes de entrenamiento**: ~4,293 observaciones
- **Imágenes de validación**: ~500 observaciones  
- **Imágenes de test**: ~1,911 observaciones
- **Formato**: JPG con múltiples resoluciones
- **Captions**: Descripciones textuales en inglés

### Descarga del Dataset

Debido al tamaño del conjunto de datos proporcionado, se debe descargar desde el siguiente enlace: https://www.kaggle.com/competitions/fungi-clef-2025/data

Se espera que la estructura de carpetas sea idéntica a la sugerida en el siguiente punto para que funcione correctamente.

## 📁 Estructura Esperada

```
dataset/
├── metadata/
│   └── FungiTastic-FewShot/
│       ├── FungiTastic-FewShot-Train.csv
│       ├── FungiTastic-FewShot-Val.csv
│       └── FungiTastic-FewShot-Test.csv
├── images/
│   └── FungiTastic-FewShot/
│       ├── train/
│       ├── val/
│       └── test/
└── captions/
    ├── train/
    ├── val/
    └── test/
```

## 🔧 Uso en el Pipeline

```python
from app.pipeline import MultimodalFungiCLEF2025Pipeline

# Inicializar pipeline con dataset
pipeline = MultimodalFungiCLEF2025Pipeline(".")

# Cargar datos
train_data = pipeline.load_data(split='train')
test_data = pipeline.load_data(split='test')
```
