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

## 📥 Descarga del Dataset

### Opción 1: FungiCLEF 2025 Oficial
1. Registrarse en ImageCLEF 2025: https://www.imageclef.org/2025/lifeclef/fungi
2. Descargar desde el portal oficial
3. Extraer en la carpeta `dataset/`

### Opción 2: Para Testing Rápido
```bash
# Crear estructura mínima para probar la aplicación
mkdir dataset\metadata\FungiTastic-FewShot
mkdir dataset\images\FungiTastic-FewShot\test\720p
mkdir dataset\captions\test
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
