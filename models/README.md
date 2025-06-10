# 🧠 Modelos Fine-tuneados

## Modelo Principal: BioCLIP Multimodal Fine-tuneado

### 📄 Descripción
- **Archivo**: `bioclip_multimodal_finetuned.pt`
- **Tamaño**: ~500-600 MB
- **Arquitectura**: BioCLIP + Adapter Multimodal
- **Especies**: 2,427 especies de hongos

### 📊 Rendimiento
- **Recall@5**: ~65-70% en validación
- **Dataset**: FungiCLEF 2025 FungiTastic-FewShot
- **Fine-tuning**: 10 epochs con fusión multimodal (beta=0.75)

## 📥 Descarga del Modelo

### Opción 1: Git LFS (Recomendado)
```bash
# Instalar Git LFS
git lfs install

# Clonar repositorio (incluye modelo)
git clone https://github.com/tu-usuario/fungiclef-2025-tfg.git
cd fungiclef-2025-tfg

# Verificar que el modelo se descargó
ls -lh models/bioclip_multimodal_finetuned.pt
```

### Opción 2: Descarga Directa
Si Git LFS no funciona:

1. **Descargar desde releases del repositorio**
2. **Colocar en**: `models/bioclip_multimodal_finetuned.pt`
3. **Verificar tamaño**: Debe ser ~500-600 MB

## 📋 Información Técnica

### Arquitectura del Adapter
```
BioCLIP Features (512D) + Texto (512D) 
    ↓
Fusión Multimodal (beta=0.75)
    ↓
Linear(512 -> 2048) + LayerNorm + GELU + Dropout(0.3)
    ↓
Linear(2048 -> 1024) + LayerNorm + GELU + Dropout(0.4)
    ↓
Linear(1024 -> 2427 classes)
```

### Parámetros de Fine-tuning
- **Learning Rate**: 3e-5 (modelo base), 2e-4 (adapter)
- **Epochs**: 10
- **Batch Size**: 48
- **Beta (fusión)**: 0.75 (75% imagen + 25% texto)
- **Loss**: OptimizedFocalLoss(alpha=0.3, gamma=1.5)

### Datos de Entrenamiento
- **Observaciones**: 4,293 imágenes
- **Especies únicas**: 2,427
- **Augmentación**: Horizontal/vertical flip, rotación, color jitter
- **Resoluciones**: 300p, 500p, 720p, fullsize

## ⚠️ Resolución de Problemas

### Error: "Modelo no encontrado"
```bash
# Verificar ubicación
dir models
# Debe existir: bioclip_multimodal_finetuned.pt
```

### Error: "CUDA out of memory"
```python
# Usar CPU en lugar de GPU
import torch
torch.cuda.empty_cache()
```
