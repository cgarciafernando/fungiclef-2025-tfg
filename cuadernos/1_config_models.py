"""
1. Configuración y Carga de Modelos
Pipeline Multimodal FungiCLEF 2025

Este archivo contiene:
- Importación de librerías
- Configuración inicial del pipeline
- Carga del ensemble de modelos (BioCLIP + DINOv2)
- Inicialización de características de hongos
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import open_clip
from tqdm.auto import tqdm
from pathlib import Path
import logging
import faiss
from scipy.spatial.distance import cosine
import warnings
import time
import re
from collections import Counter, defaultdict
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# Configuración de logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fungiclef_multimodal_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuraciones globales
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MultimodalFungiCLEF2025Pipeline:
    """
    Pipeline optimizado para FungiCLEF 2025 usando ensemble de modelos,
    procesamiento de captions y clasificación jerárquica con procesamiento multimodal.
    """
    def __init__(self, base_path, metadata_subdir='metadata/FungiTastic-FewShot',
                 image_subdir='images/FungiTastic-FewShot', caption_subdir='captions'):

        # Rutas de directorios siguiendo la estructura del repositorio
        self.base_path = Path(base_path)
        self.metadata_path = self.base_path / metadata_subdir
        self.image_path = self.base_path / image_subdir
        self.caption_path = self.base_path / caption_subdir
    
        # Configuraciones básicas
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Target resolutions
        self.target_resolutions = ['300p', '500p', '720p', 'fullsize']
        self.resolution_weights = {'300p': 0.6, '500p': 0.9, '720p': 1.1, 'fullsize': 1.2}
    
        # Configuración del ensemble de modelos
        self.model_ensemble_weights = {'bioclip': 1.4, 'dinov2': 1.2, 'efficientnet': 0.6}
        
        # Cargar modelos para ensemble
        self.models = self._load_model_ensemble()
        if not self.models:
            raise RuntimeError("No se pudieron cargar los modelos. Abortando pipeline.")
    
        # Parámetros optimizados
        self.k_neighbors = 45
        self.k_search_multiplier = 12
        self.ensemble_weights = {'knn': 0.45, 'centroid': 0.25, 'medoid': 0.15, 'metadata': 0.4, 'captions': 0.45}
        self.metadata_sim_weights = {'habitat': 0.5, 'substrate': 0.4, 'bioregion': 0.2, 'month': 0.3}
        
        # Boost de especies raras
        self.use_rare_species_boost = True
        self.rare_species_boost_factor = 0.35
        self.rare_species_threshold = 10
        
        # Parámetros HNSW
        self.hnsw_m = 160
        self.hnsw_ef_construction = 400
        self.hnsw_ef_search = 350
        
        # Parámetros para augmentación
        self.use_test_augmentation = True
        self.test_augmentation_methods = ['original', 'flip_h', 'flip_v', 'rotate_90', 'rotate_270', 'color_jitter', 'blur']
        
        # Parámetros para taxonomía y captions
        self.taxonomy_level_weights = {
            'genus': 0.45,
            'family': 0.25,
            'order': 0.12,
            'class': 0.04,
            'phylum': 0.01
        }
        
        # Parámetros para procesamiento avanzado de captions
        self.use_advanced_caption_processing = True
        self.caption_feature_extraction = True
        self.caption_taxonomic_boost = True
        self.caption_hierarchical_boost = True
        
        # Parámetros para procesamiento multimodal
        self.use_multimodal_processing = True
        self.text_image_weight = 0.35  # Peso relativo del texto vs imagen (0.35 texto, 0.65 imagen)
        
        # Inicializar términos y características
        self._init_fungi_features()
        
        # Almacenamiento interno
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.num_classes = 0
        self.train_data_cache = None
        self.val_data_cache = None
        self.index_data_cache = None
        self.class_frequency = None
        self.caption_term_frequencies = {}
        self.caption_taxonomic_mapping = {}
        self.species_to_taxonomy = {}
        self.taxonomy_to_species = {}
        
        # Para modelo ecológico contextual
        self.habitat_species = defaultdict(Counter)
        self.substrate_species = defaultdict(Counter)
        self.region_species = defaultdict(Counter)
        self.month_species = defaultdict(Counter)
        self.habitat_counts = Counter()
        self.substrate_counts = Counter()
        self.region_counts = Counter()
        self.month_counts = Counter()
        self.habitat_probs = defaultdict(dict)
        self.substrate_probs = defaultdict(dict)
        self.region_probs = defaultdict(dict)
        self.month_probs = defaultdict(dict)
        
        logger.info("Pipeline multimodal inicializado con ensemble de modelos")

    def _init_fungi_features(self):
        """Inicializa términos y características para procesamiento de captions"""
        # Lista de términos para características clave de hongos
        self.fungi_feature_terms = {
            'color': ['white', 'brown', 'yellow', 'red', 'orange', 'purple', 'black', 'blue', 'green', 'pink', 'gray', 'beige', 'cream', 'golden', 'tan', 'ivory'],
            'shape': ['cap', 'stem', 'stipe', 'gills', 'pores', 'ring', 'volva', 'veil', 'fruiting body', 'conical', 'convex', 'flat', 'depressed', 'funnel', 'club', 'bell'],
            'texture': ['smooth', 'scaly', 'fibrous', 'slimy', 'sticky', 'velvety', 'powdery', 'wrinkled', 'gelatinous', 'leathery', 'brittle', 'rough', 'bumpy', 'furry'],
            'habitat': ['forest', 'wood', 'grass', 'meadow', 'soil', 'compost', 'mulch', 'tree', 'pine', 'oak', 'beech', 'birch', 'eucalyptus', 'tropical', 'temperate', 'alpine']
        }
        
        # Términos taxonómicos relevantes
        self.taxonomic_terms = ['genus', 'family', 'order', 'class', 'phylum', 'kingdom', 'amanita', 'boletus', 'russula', 'lactarius', 'cortinarius', 'agaric', 'bolete', 'polypore', 'basidiomycota', 'ascomycota']
        
        # Diccionario para jerarquía taxonómica
        self.taxonomic_hierarchy = {
            'kingdom': ['fungi'],
            'phylum': ['basidiomycota', 'ascomycota'],
            'class': ['agaricomycetes', 'sordariomycetes', 'dothideomycetes', 'leotiomycetes', 'pezizomycetes'],
            'order': ['agaricales', 'boletales', 'polyporales', 'russulales', 'hymenochaetales', 'pezizales'],
            'family': ['amanitaceae', 'boletaceae', 'russulaceae', 'polyporaceae', 'cortinariaceae', 'hygrophoraceae'],
            'genus': ['amanita', 'boletus', 'russula', 'lactarius', 'cortinarius', 'agaricus', 'pleurotus', 'marasmius', 'mycena', 'ganoderma']
        }
        
        # Características de hongos por pares (para mayor especificidad)
        self.fungi_feature_pairs = {
            'color_location': [
                'white cap', 'brown cap', 'yellow cap', 'red cap',
                'white stem', 'brown stem', 'yellow stem',
                'white gills', 'brown gills', 'black gills', 'pink gills'
            ],
            'texture_location': [
                'smooth cap', 'scaly cap', 'wrinkled cap',
                'fibrous stem', 'smooth stem', 'velvety cap'
            ]
        }
        
        # Pesos de contexto ecológico
        self.context_weights = {
            'habitat': 0.45,
            'substrate': 0.35,
            'region': 0.12,
            'month': 0.25
        }

    def _load_model_ensemble(self):
        """Carga múltiples modelos para ensemble"""
        models = {}
        
        # 1. BioCLIP Base (modelo principal)
        logger.info("Cargando modelo BioCLIP base...")
        try:
            bioclip_model_name = 'hf-hub:imageomics/bioclip'
            bioclip_model, _, bioclip_preprocess = open_clip.create_model_and_transforms(
                bioclip_model_name,
                pretrained=None
            )
            bioclip_tokenizer = open_clip.get_tokenizer(bioclip_model_name)
            
            bioclip_model.to(self.device)
            bioclip_model.eval()
            
            models['bioclip'] = {
                'model': bioclip_model,
                'preprocess': bioclip_preprocess,
                'tokenizer': bioclip_tokenizer,
                'weight': self.model_ensemble_weights['bioclip']
            }
            logger.info("BioCLIP base cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error cargando BioCLIP: {e}", exc_info=True)
        
        # 2. DINOv2
        logger.info("Cargando modelo DINOv2...")
        try:
            dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            dinov2_model.to(self.device)
            dinov2_model.eval()
            
            # Transformación para DINOv2
            dinov2_transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            
            models['dinov2'] = {
                'model': dinov2_model,
                'preprocess': dinov2_transform,
                'weight': self.model_ensemble_weights['dinov2']
            }
            logger.info("DINOv2 cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error cargando DINOv2 (no crítico): {e}")
        
        if not models:
            raise RuntimeError("No se pudo cargar ningún modelo.")
        
        logger.info(f"Cargados {len(models)} modelos para ensemble.")
        return models