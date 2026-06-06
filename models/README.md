# Model
 
## BioCLIP Multimodal Fine-tuned
 
**File:** `bioclip_multimodal_finetuned.pt`
**Size:** ~500-600 MB
**Architecture:** BioCLIP + Multimodal Adapter
**Species:** 2,427 fungal species
 
The model was not uploaded to this repository due to file size. Download it from the releases page or contact the author.
 
## Architecture
 
```
BioCLIP Features (512D) + Text (512D)
    ↓
Multimodal fusion (β=0.75)
    ↓
Linear(512 → 2048) + LayerNorm + GELU + Dropout(0.3)
    ↓
Linear(2048 → 1024) + LayerNorm + GELU + Dropout(0.4)
    ↓
Linear(1024 → 2427 classes)
```
 
## Training parameters
 
| Parameter | Value |
|---|---|
| Learning rate (backbone) | 3e-5 |
| Learning rate (adapter) | 2e-4 |
| Epochs | 10 |
| Batch size | 48 |
| Fusion beta | 0.75 (75% image + 25% text) |
| Loss | Focal Loss (α=0.3, γ=1.5) |
| Training observations | 4,293 |
 
## Troubleshooting
 
**Model not found**
```bash
dir models
# bioclip_multimodal_finetuned.pt must exist in this directory
```
 
**CUDA out of memory**
```python
import torch
torch.cuda.empty_cache()
```
