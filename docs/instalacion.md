# 📘 Guía de Instalación

## 🔧 Requisitos del Sistema

### Software Necesario
- **Python**: 3.8 o superior (recomendado 3.11)
- **Git**: Para clonar el repositorio
- **Git LFS**: Para descargar modelos grandes

### Hardware Recomendado
- **RAM**: 8GB mínimo (16GB recomendado)
- **Almacenamiento**: 10GB libres
- **GPU**: CUDA compatible (opcional, mejora rendimiento)

## ⚡ Instalación Rápida

### 1. Clonar Repositorio
```bash
git clone https://github.com/tu-usuario/fungiclef-2025-tfg.git
cd fungiclef-2025-tfg
```

### 2. Crear Entorno Virtual (Recomendado)
```bash
# Python venv
python -m venv ml_env
ml_env\Scripts\activate  # Windows
source ml_env/bin/activate  # Linux/Mac

# O con conda
conda create -n fungiclef python=3.11
conda activate fungiclef
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Git LFS
```bash
git lfs install
git lfs pull  # Descargar modelo fine-tuneado
```

### 5. Verificar Instalación
```bash
python app/main.py
# Debe abrir interfaz web en http://localhost:7860
```

## 🔍 Resolución de Problemas

### Error: "No module named torch"
```bash
# Instalar PyTorch específico para tu sistema
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# O para GPU CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Error: "CUDA out of memory"
```python
# En app/pipeline.py, cambiar device a CPU
self.device = torch.device('cpu')
```

### Error: "Modelo no encontrado"
```bash
# Verificar descarga con Git LFS
ls -la models/
git lfs pull
```
