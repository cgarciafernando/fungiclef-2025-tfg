![fungiclef-2025-tfg](https://socialify.git.ci/cgarciafernando/fungiclef-2025-tfg/image?font=Inter&language=1&name=1&owner=1&pattern=Circuit+Board&stargazers=1&theme=Dark)
 
# FungiCLEF 2025 — Multimodal Pipeline for Rare Fungal Species Classification
 
<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue.svg" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/BioCLIP-finetuned-green.svg" />
  <img src="https://img.shields.io/badge/CLEF_2025-Published-blueviolet.svg" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
</p>
<p align="center">
  <strong>Ranked 22nd out of 74 international teams &nbsp;·&nbsp; Recall@5: 0.574 &nbsp;·&nbsp; +28% above competition median</strong>
</p>
Bachelor's thesis project at Universidad de Huelva. A multimodal classification pipeline for rare fungal species identification, combining Vision-Language Models, ecological context, and taxonomic knowledge to handle extreme data scarcity (84.6% of species with fewer than 5 training samples).
 
> **Paper:** [I2C-UHU-PEGASUS at FungiCLEF 2025: Multimodal Pipeline for Rare Fungal Species Classification Using Fine-Tuned VLMs and Ecological Context](https://ceur-ws.org/Vol-4038/paper_236.pdf)
> Fernando Carrillo García, Victoria Pachón Álvarez, Jacinto Mata Vázquez, Manuel Guerrero García — University of Huelva, Spain
 
---
 
## What does this project do?
 
The pipeline combines multiple information sources to identify fungal species from images and metadata. Rather than relying on visual features alone, it integrates morphological descriptions, ecological context (habitat, substrate, biogeographic region, season), and hierarchical taxonomic information to improve classification in few-shot scenarios.
 
**Key components:**
- Fine-tuned BioCLIP with multimodal classifier (visual + textual fusion, β=0.75)
- DINOv2 ensemble for complementary visual representations
- Probabilistic ecological context model across 4 dimensions
- Multi-strategy fusion: k-NN, centroid, medoid, metadata matching, description similarity
- FAISS with HNSW indexing for efficient similarity search
- Adaptive weighted sampling and rare-class boosting for class imbalance
## Results
 
| Metric | Value |
|---|---|
| Recall@5 (test set) | 0.574 |
| Competition rank | 22nd / 74 teams |
| Above median | +28% |
| Total improvement over baseline | +8.1% |
| Species classified | 2,427 |
 
## Project structure
 
```
fungiclef-2025-tfg/
├── app/
│   ├── main.py            # Gradio interface
│   ├── pipeline.py        # Multimodal pipeline
│   └── utils.py
├── dataset/               # Download instructions
├── predicciones/          # Prediction outputs
├── resultados/            # Metrics and analysis
├── models/
│   └── bioclip_multimodal_finetuned.pt
└── docs/
```
 
## How to run it
 
**1. Clone the repository**
```bash
git clone https://github.com/cgarciafernando/fungiclef-2025-tfg
cd fungiclef-2025-tfg
```
 
**2. Install dependencies**
```bash
pip install -r requirements.txt
```
 
**3. Download the fine-tuned model**
 
See instructions in [`models/README.md`](models/README.md).
 
**4. Run the demo**
```bash
python app/main.py
```
 
## Author
 
**Fernando Carrillo García**
B.S. Computer Engineering — Universidad de Huelva
