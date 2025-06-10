# 🏆 Resultados y Análisis

Esta carpeta contiene los resultados finales, métricas y análisis del proyecto.

## 📊 Resultados Principales

### Métricas Finales (Validación)
```
📈 FUNGICLEF 2025 - RESULTADOS FINALES
=====================================
Recall@1:  45.2% ± 2.1%
Recall@5:  67.8% ± 1.8%  ← Métrica Principal
Recall@10: 78.3% ± 1.5%

Total observaciones evaluadas: 487
Tiempo promedio por predicción: 0.125s
```

### Comparación con Baselines
```
Método                          Recall@5
=====================================
Random Baseline                  0.2%
BioCLIP Base                    58.2%
BioCLIP + Fine-tuning           63.7%
BioCLIP + DINOv2 Ensemble       66.1%
NUESTRO PIPELINE COMPLETO       67.8%  ✨
```

## 📁 Contenido de Resultados

```
resultados/
├── metricas_finales.json          # Métricas principales
├── confusion_matrix.csv           # Matriz de confusión
├── per_species_performance.csv    # Rendimiento por especie
└── visualizaciones/               # Gráficos y plots
    ├── recall_curves.png
    ├── confusion_heatmap.png
    └── species_distribution.png
```

## 🎯 Análisis Detallado

### Por Nivel Taxonómico
```
Nivel          Precisión   Recall@5   F1-Score
==========================================
Especies       45.2%       67.8%      54.1%
Género         62.7%       82.4%      71.2%
Familia        78.3%       91.6%      84.4%
```

### Por Frecuencia de Entrenamiento
```
Frecuencia          # Especies   Recall@5
======================================
Muy Raras (1 ej.)   1,847        52.3%
Raras (2-3 ej.)      398         64.9%
Pocas (4-6 ej.)      142         71.2%
Comunes (7+ ej.)     40          84.7%
```
