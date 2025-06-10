# 🎮 Manual de Uso

## 🚀 Inicio Rápido

### Lanzar la Aplicación
```bash
# Activar entorno virtual
ml_env\Scripts\activate  # Windows

# Ejecutar aplicación
python app/main.py

# Abrir navegador en: http://localhost:7860
```

## 🔍 Uso de la Interfaz Web

### 1. Selección de Imagen
- Usar el dropdown para elegir una imagen de test
- Las imágenes con 📝 tienen descripción textual

### 2. Análisis
- Hacer clic en "🔮 Analizar"
- El sistema procesará la imagen y generará predicciones

### 3. Resultados
- **Top 5 Especies**: Predicciones con confianza
- **Galería Visual**: Ejemplos de especies similares  
- **Espacio Vectorial**: Visualización 2D de embeddings
- **Información Detallada**: Taxonomía y metadatos

## 📊 Interpretación de Resultados

### Scores de Confianza
- **0.8-1.0**: Muy alta confianza
- **0.6-0.8**: Alta confianza  
- **0.4-0.6**: Confianza media
- **0.2-0.4**: Baja confianza
- **0.0-0.2**: Muy baja confianza

### Visualización del Espacio Vectorial
- **⭐ Verde**: Imagen analizada
- **🔴 Rojos**: Top predicciones
- **🔵 Azules**: Especies similares
- **Clusters**: Agrupaciones de especies relacionadas

## 🛠️ Uso Avanzado

### Análisis de Dataset Propio
```python
from app.pipeline import MultimodalFungiCLEF2025Pipeline

# Inicializar pipeline
pipeline = MultimodalFungiCLEF2025Pipeline(".")

# Cargar tus datos
custom_data = pipeline.load_data(split='custom')

# Procesar y predecir
predictions = pipeline.predict_observation(obs_data, train_data, index_data)
```

### Entrenamiento con Datos Adicionales
```python
# Añadir más datos de entrenamiento
new_train_data = pipeline.load_data(split='new_train')

# Re-entrenar modelo
pipeline.finetune_bioclip_multimodal(new_train_data, val_data)
```

## 🎯 Tips y Mejores Prácticas

### Para Mejores Resultados
1. **Calidad de imagen**: Usar resoluciones altas (720p+)
2. **Información textual**: Incluir descripciones detalladas
3. **Contexto ecológico**: Proporcionar hábitat y sustrato
4. **Múltiples ángulos**: Analizar varias fotos de la misma especie

### Limitaciones a Considerar
- **Especies raras**: Menor precisión con poccos ejemplos
- **Especies similares**: Posible confusión entre géneros cercanos
- **Calidad de imagen**: Imágenes borrosas afectan precisión
- **Contexto incompleto**: Falta de metadatos reduce precisión
