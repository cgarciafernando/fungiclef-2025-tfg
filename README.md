# 🍄 FungiCLEF 2025 - Sistema de Clasificación de Hongos

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/gradio-4.0+-orange.svg)](https://gradio.app/)
[![BioCLIP](https://img.shields.io/badge/BioCLIP-finetuned-green.svg)](https://github.com/imageomics/bioclip)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Descripción

**Trabajo de Fin de Grado** - Sistema avanzado de clasificación automática de especies de hongos basado en:

- 🧠 **Pipeline Multimodal**: Combina información visual y textual
- 🔬 **BioCLIP Fine-tuned**: Modelo especializado en biodiversidad adaptado para hongos
- 🤖 **DINOv2 Ensemble**: Características visuales complementarias
- 📊 **FAISS Similarity Search**: Búsqueda eficiente en espacio vectorial
- 🌐 **Interfaz Gradio**: Demo interactiva web

## 📁 Estructura del Proyecto

```
fungiclef-2025-tfg/
├── app/                    # 🎯 Aplicación principal
│   ├── main.py            # Interfaz Gradio
│   ├── pipeline.py        # Pipeline multimodal
│   └── utils.py           # Utilidades
├── dataset/               # 📊 Datos (instrucciones de descarga)
├── predicciones/          # 📈 Resultados de predicción
├── resultados/            # 🏆 Métricas y análisis
├── models/                # 🧠 Modelos entrenados
│   └── bioclip_multimodal_finetuned.pt
└── docs/                  # 📖 Documentación
```

## ⚡ Instalación Rápida

### 1. Clonar repositorio
```bash
git clone https://github.com/tu-usuario/fungiclef-2025-tfg.git
cd fungiclef-2025-tfg
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Descargar modelo fine-tuneado
Ver instrucciones en [`models/README.md`](models/README.md)

### 4. Ejecutar demo
```bash
python app/main.py
```

## 🎯 Características Principales

### Pipeline Multimodal
- **Fusión de información**: Combina características visuales (BioCLIP + DINOv2) y textuales
- **Fine-tuning especializado**: Modelo adaptado específicamente para hongos
- **Búsqueda por similitud**: FAISS para retrieval eficiente de especies similares

### Interfaz Intuitiva
- **Exploración visual**: Visualización del espacio vectorial 2D con clusters
- **Predicciones explicables**: Top-5 especies con confianza y ejemplos visuales
- **Información taxonómica**: Detalles de género, familia y características ecológicas

## 📊 Resultados

- **Recall@5**: 0.574 sobre el conjunto de prueba. Posición 22 de 74 en la tabla de clasificación.
- **Especies clasificadas**: 2,427 especies únicas
- **Dataset**: FungiCLEF 2025 FungiTastic-FewShot

## 🛠️ Tecnologías Utilizadas

- **Deep Learning**: PyTorch, Transformers
- **Modelos**: BioCLIP, DINOv2, CLIP
- **Búsqueda**: FAISS, scikit-learn
- **Interface**: Gradio, Matplotlib
- **Deploy**: Hugging Face Spaces

## 📖 Documentación

- 📘 [Guía de Instalación](docs/instalacion.md)
- 🎮 [Manual de Uso](docs/uso.md)

## 👨‍🎓 Autor

**Fernando Carrillo García**  
Trabajo de Fin de Grado - Universidad de Huelva, 
Grado en Ingeniería Informática

## 📜 Licencia

Este proyecto está bajo la Licencia MIT - ver [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- [LifeCLEF](https://www.imageclef.org/LifeCLEF2025) por el dataset FungiCLEF 2025
- [Imageomics](https://github.com/imageomics/bioclip) por BioCLIP
- [Meta AI](https://github.com/facebookresearch/dinov2) por DINOv2
