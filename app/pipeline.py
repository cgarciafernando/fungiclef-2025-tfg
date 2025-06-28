"""
Pipeline Multimodal con EARLY FUSION para FungiCLEF 2025
Combina BioCLIP + DINOv2 mediante concatenación de embeddings
CON PARÁMETROS OPTIMIZADOS SEGÚN MEMORIA
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
import math
from collections import Counter, defaultdict
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Configuración de logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MultimodalFungiCLEF2025Pipeline:
    """
    Pipeline con EARLY FUSION: concatena BioCLIP + DINOv2 embeddings
    CON PARÁMETROS OPTIMIZADOS SEGÚN MEMORIA
    """
    def __init__(self, base_path, metadata_subdir='dataset/metadata/FungiTastic-FewShot',
                 image_subdir='dataset/images/FungiTastic-FewShot', caption_subdir='dataset/captions'):

        # Rutas de directorios siguiendo la estructura del repositorio
        self.base_path = Path(base_path)
        self.metadata_path = self.base_path / metadata_subdir
        self.image_path = self.base_path / image_subdir
        self.caption_path = self.base_path / caption_subdir
    
        # Configuraciones básicas
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Usando dispositivo: {self.device}")
        
        # === PARÁMETROS OPTIMIZADOS SEGÚN MEMORIA ===
        
        # Target resolutions con pesos optimizados
        self.target_resolutions = ['300p', '500p', '720p', 'fullsize']
        self.resolution_weights = {
            '300p': 0.4, 
            '500p': 0.9, 
            '720p': 1.3,      # Valor optimizado según memoria
            'fullsize': 1.5   # Valor optimizado según memoria
        }
    
        self.bioclip_weight = 0.7  # 70% BioCLIP (especialista biología)
        self.dinov2_weight = 0.3   # 30% DINOv2 (características visuales generales)
        
        # Cargar modelos para ensemble
        self.models = self._load_model_ensemble()
        if not self.models:
            raise RuntimeError("No se pudieron cargar los modelos. Abortando pipeline.")
    
        # === HIPERPARÁMETROS OPTIMIZADOS SEGÚN MEMORIA ===
        self.k_neighbors = 50                    # Optimizado
        self.k_search_multiplier = 15           # Optimizado
        self.ensemble_weights = {               # Pesos optimizados
            'knn': 0.5, 
            'centroid': 0.3, 
            'medoid': 0.2, 
            'metadata': 0.45, 
            'captions': 0.5
        }
        self.metadata_sim_weights = {           # Pesos de similitud optimizados
            'habitat': 0.5, 
            'substrate': 0.4, 
            'bioregion': 0.2, 
            'month': 0.3
        }
        
        # Boost de especies raras optimizado
        self.use_rare_species_boost = True
        self.rare_species_boost_factor = 0.5    # Optimizado según memoria
        self.rare_species_threshold = 15        # Optimizado según memoria
        
        # Parámetros HNSW optimizados
        self.hnsw_m = 160
        self.hnsw_ef_construction = 400
        self.hnsw_ef_search = 400               # Optimizado según memoria
        
        # Parámetros para taxonomía y captions optimizados
        self.taxonomy_level_weights = {
            'genus': 0.45,
            'family': 0.25,
            'order': 0.12,
            'class': 0.04,
            'phylum': 0.01
        }
        
        # Parámetros para procesamiento multimodal optimizados
        self.use_multimodal_processing = True
        self.text_image_weight = 0.2            # Optimizado según memoria (20% texto)
        
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
        
        # Para modelo ecológico contextual con pesos optimizados
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
        
        # Pesos de contexto ecológico optimizados según memoria
        self.context_weights = {
            'habitat': 0.45,     # Optimizado
            'substrate': 0.35,   # Optimizado  
            'region': 0.12,      # Optimizado
            'month': 0.25        # Optimizado
        }
        
        logger.info("Pipeline con EARLY FUSION inicializado con parámetros optimizados")

    def _init_fungi_features(self):
        """Inicializa términos y características para procesamiento de captions"""
        self.fungi_feature_terms = {
            'color': ['white', 'brown', 'yellow', 'red', 'orange', 'purple', 'black', 'blue', 'green', 'pink', 'gray', 'beige', 'cream', 'golden', 'tan', 'ivory'],
            'shape': ['cap', 'stem', 'stipe', 'gills', 'pores', 'ring', 'volva', 'veil', 'fruiting body', 'conical', 'convex', 'flat', 'depressed', 'funnel', 'club', 'bell'],
            'texture': ['smooth', 'scaly', 'fibrous', 'slimy', 'sticky', 'velvety', 'powdery', 'wrinkled', 'gelatinous', 'leathery', 'brittle', 'rough', 'bumpy', 'furry'],
            'habitat': ['forest', 'wood', 'grass', 'meadow', 'soil', 'compost', 'mulch', 'tree', 'pine', 'oak', 'beech', 'birch', 'eucalyptus', 'tropical', 'temperate', 'alpine']
        }
        
        self.taxonomic_terms = ['genus', 'family', 'order', 'class', 'phylum', 'kingdom', 'amanita', 'boletus', 'russula', 'lactarius', 'cortinarius', 'agaric', 'bolete', 'polypore', 'basidiomycota', 'ascomycota']
        
        self.taxonomic_hierarchy = {
            'kingdom': ['fungi'],
            'phylum': ['basidiomycota', 'ascomycota'],
            'class': ['agaricomycetes', 'sordariomycetes', 'dothideomycetes', 'leotiomycetes', 'pezizomycetes'],
            'order': ['agaricales', 'boletales', 'polyporales', 'russulales', 'hymenochaetales', 'pezizales'],
            'family': ['amanitaceae', 'boletaceae', 'russulaceae', 'polyporaceae', 'cortinariaceae', 'hygrophoraceae'],
            'genus': ['amanita', 'boletus', 'russula', 'lactarius', 'cortinarius', 'agaricus', 'pleurotus', 'marasmius', 'mycena', 'ganoderma']
        }
        
        self.fungi_feature_pairs = {
            'color_location': ['white cap', 'brown cap', 'yellow cap', 'red cap', 'white stem', 'brown stem', 'yellow stem', 'white gills', 'brown gills', 'black gills', 'pink gills'],
            'texture_location': ['smooth cap', 'scaly cap', 'wrinkled cap', 'fibrous stem', 'smooth stem', 'velvety cap']
        }

    def _load_model_ensemble(self):
        """Carga BioCLIP + DINOv2 para early fusion"""
        models = {}
        
        # 1. BioCLIP Base (modelo principal)
        logger.info("Cargando modelo BioCLIP base...")
        try:
            bioclip_model_name = 'hf-hub:imageomics/bioclip'
            bioclip_model, _, bioclip_preprocess = open_clip.create_model_and_transforms(
                bioclip_model_name, pretrained=None
            )
            bioclip_tokenizer = open_clip.get_tokenizer(bioclip_model_name)
            
            bioclip_model.to(self.device)
            bioclip_model.eval()
            
            models['bioclip'] = {
                'model': bioclip_model,
                'preprocess': bioclip_preprocess,
                'tokenizer': bioclip_tokenizer,
                'finetuned': False
            }
            logger.info("BioCLIP base cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error cargando BioCLIP: {e}")
            raise
        
        # 2. DINOv2 (REQUERIDO para early fusion)
        logger.info("Cargando modelo DINOv2...")
        try:
            dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            dinov2_model.to(self.device)
            dinov2_model.eval()
            
            dinov2_transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            
            models['dinov2'] = {
                'model': dinov2_model,
                'preprocess': dinov2_transform
            }
            logger.info("DINOv2 cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error cargando DINOv2: {e}")
            raise RuntimeError("DINOv2 es REQUERIDO para early fusion")
        
        logger.info(f"Cargados {len(models)} modelos para early fusion.")
        return models

    def _concatenate_embeddings(self, bioclip_embed, dinov2_embed):
        """
        Early Fusion: Concatena BioCLIP (~512D) + DINOv2 (~768D) = ~1280D
        Con normalización robusta y pesos adaptativos para manejar espacios semánticos diferentes
        """
        # 1. Normalizar cada embedding individualmente (L2 norm)
        bioclip_norm = np.linalg.norm(bioclip_embed)
        dinov2_norm = np.linalg.norm(dinov2_embed)
        
        # Evitar división por cero
        bioclip_normalized = bioclip_embed / (bioclip_norm + 1e-8)
        dinov2_normalized = dinov2_embed / (dinov2_norm + 1e-8)
        
        # 2. Aplicar pesos adaptativos
        # BioCLIP tiene mayor peso porque está alineado semánticamente para biología
        alpha = self.bioclip_weight  # 0.7 = 70% BioCLIP
        beta = self.dinov2_weight    # 0.3 = 30% DINOv2
        
        bioclip_weighted = bioclip_normalized * alpha
        dinov2_weighted = dinov2_normalized * beta
        
        # 3. Concatenar embeddings normalizados y ponderados
        # Resultado: ~1280D (512*0.7 + 768*0.3 concatenados)
        concatenated = np.concatenate([bioclip_weighted, dinov2_weighted], axis=-1)
        
        # 4. Normalización final para compatibilidad con FAISS cosine similarity
        final_norm = np.linalg.norm(concatenated)
        if final_norm > 1e-8:
            concatenated = concatenated / final_norm
        else:
            logger.warning("⚠️ Embedding concatenado con norma casi cero")
            # Retornar vector normalizado por defecto
            concatenated = concatenated / (final_norm + 1e-8)
        
        return concatenated.astype('float32')

    def load_data(self, split='train'):
        """Cargar datos (train, val, o test) desde archivos CSV"""
        logger.info(f"Cargando datos {split} desde {self.metadata_path}...")
        
        if split == 'train': 
            metadata_filename = 'FungiTastic-FewShot-Train.csv'
        elif split == 'val': 
            metadata_filename = 'FungiTastic-FewShot-Val.csv'
        elif split == 'test': 
            metadata_filename = 'FungiTastic-FewShot-Test.csv'
        else: 
            raise ValueError("Split debe ser 'train', 'val', o 'test'")
        
        metadata_file_path = self.metadata_path / metadata_filename
        if not metadata_file_path.is_file():
            logger.error(f"Archivo de metadatos no encontrado en {metadata_file_path}")
            return {'observations': {}}
        
        logger.info(f"Leyendo archivo CSV: {metadata_file_path}")
        try:
            metadata_df = pd.read_csv(metadata_file_path, low_memory=False)
        except Exception as e:
            logger.error(f"Error al leer el archivo CSV {metadata_file_path}: {e}")
            return {'observations': {}}
        
        # Class mapping for train and val
        if split == 'train' or split == 'val':
            required_cols_train_val = ['observationID', 'filename', 'category_id']
            if not all(col in metadata_df.columns for col in required_cols_train_val):
                logger.error(f"Missing required columns in {metadata_file_path} for split '{split}'. Aborting.")
                return {'observations': {}}
            
            metadata_df['category_id'] = pd.to_numeric(metadata_df['category_id'], errors='coerce').fillna(-1).astype(int)
            
            if split == 'train':
                logger.info("Building class mapping from training data...")
                valid_category_mask = metadata_df['category_id'] != -1
                unique_classes = sorted(metadata_df[valid_category_mask]['category_id'].unique())
                
                self.class_to_idx.clear()
                self.idx_to_class.clear()
                self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
                self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
                self.num_classes = len(self.class_to_idx)
                
                class_counts = metadata_df['category_id'].value_counts().to_dict()
                self.class_frequency = {cls_id: class_counts.get(cls_id, 0) for cls_id in unique_classes}
        
        elif split == 'test':
            required_cols_test = ['observationID', 'filename']
            if not all(col in metadata_df.columns for col in required_cols_test):
                logger.error(f"Missing required columns in {metadata_file_path} for 'test' split. Aborting.")
                return {'observations': {}}
        
        # Process metadata rows
        data = {'observations': {}}
        logger.info(f"Processing {len(metadata_df)} metadata rows for {split}...")
        
        taxonomic_cols = ['genus', 'family', 'order', 'class', 'phylum', 'species']
        metadata_cols = ['habitat', 'substrate', 'biogeographicalRegion', 'month']
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc=f"Processing {split}"):
            try:
                obs_id = row['observationID']
                img_filename = row['filename']
                
                if 'category_id' in row and pd.notna(row['category_id']):
                    original_class_id = int(row['category_id'])
                else:
                    original_class_id = -1
                
            except Exception as e:
                logger.warning(f"Error in row {idx+1}: {e}")
                continue
                
            if obs_id not in data['observations']:
                data['observations'][obs_id] = {
                    'original_class_id': original_class_id,
                    'image_entries': [],
                    'metadata': {}
                }
                
                # Process metadata
                try:
                    metadata = {}
                    
                    for col in metadata_cols:
                        if col in row and pd.notna(row[col]):
                            if col in ['month']:
                                metadata[col] = int(row[col]) if pd.notna(row.get(col)) else -1
                            else:
                                metadata[col] = str(row.get(col, '')).strip()
                    
                    for col in taxonomic_cols:
                        if col in row and pd.notna(row[col]):
                            tax_value = str(row[col]).strip()
                            if tax_value and tax_value != 'nan':
                                metadata[col] = tax_value
                    
                    data['observations'][obs_id]['metadata'] = metadata
                    
                except Exception as meta_e:
                    logger.warning(f"Metadata error for obs {obs_id} row {idx+1}: {meta_e}")
            
            # Process image entry
            image_entry = {'filename': img_filename, 'paths': {}, 'caption': None}
            img_found = False
            
            for res in self.target_resolutions:
                res_path = self.image_path / split / res / img_filename
                if res_path.is_file():
                    image_entry['paths'][res] = str(res_path)
                    img_found = True
            
            if not img_found:
                continue
                
            # Process captions if available
            cap_path = self.caption_path / split / f"{img_filename}.json"
            if cap_path.is_file():
                try:
                    with open(cap_path, 'r', encoding='utf-8') as f:
                        cap_data = json.load(f)
                        
                    if isinstance(cap_data, str):
                        image_entry['caption'] = cap_data
                    elif isinstance(cap_data, dict) and 'caption' in cap_data:
                        image_entry['caption'] = cap_data['caption']
                except Exception:
                    pass
                    
            if obs_id in data['observations']:
                data['observations'][obs_id]['image_entries'].append(image_entry)
                
        # Remove observations without valid images
        orig_obs_count = len(data['observations'])
        data['observations'] = {o: d for o, d in data['observations'].items() if d.get('image_entries')}
        if orig_obs_count != len(data['observations']):
            logger.info(f"Removed {orig_obs_count - len(data['observations'])} observations without images.")
            
        return data

    def _create_structured_prompt(self, obs_data):
        """Crea un prompt estructurado combinando captions y metadatos"""
        prompt_parts = []
        prompt_parts.append("Identify this fungi species.")
        
        # Combinar captions
        all_captions = []
        for img_entry in obs_data.get('image_entries', []):
            if img_entry.get('caption'):
                all_captions.append(img_entry['caption'])
        
        if all_captions:
            main_caption = max(all_captions, key=len)
            prompt_parts.append(f"Description: {main_caption}")
        
        # Añadir metadatos estructurados
        if 'metadata' in obs_data:
            metadata = obs_data['metadata']
            metadata_parts = []
            
            # Información ecológica
            ecology_info = []
            if metadata.get('habitat'):
                ecology_info.append(f"habitat: {metadata['habitat']}")
            if metadata.get('substrate'):
                ecology_info.append(f"substrate: {metadata['substrate']}")
            if metadata.get('biogeographicalRegion'):
                ecology_info.append(f"region: {metadata['biogeographicalRegion']}")
            
            # Información estacional
            if metadata.get('month', -1) != -1 and 1 <= metadata['month'] <= 12:
                months = ["January", "February", "March", "April", "May", "June", 
                         "July", "August", "September", "October", "November", "December"]
                month_name = months[metadata['month'] - 1]
                ecology_info.append(f"collected in: {month_name}")
            
            if ecology_info:
                metadata_parts.append("Ecological context: " + ", ".join(ecology_info))
            
            # Información taxonómica
            taxonomy_info = []
            for level in ['genus', 'family', 'order', 'class', 'phylum']:
                if level in metadata and metadata[level]:
                    taxonomy_info.append(f"{level}: {metadata[level]}")
            
            if taxonomy_info:
                metadata_parts.append("Taxonomic information: " + ", ".join(taxonomy_info))
            
            if metadata_parts:
                prompt_parts.append("\\n".join(metadata_parts))
        
        final_prompt = "\\n".join(prompt_parts)
        return final_prompt

    def extract_features_multimodal(self, data):
        """Extracción con EARLY FUSION: BioCLIP + DINOv2 concatenados"""
        logger.info(f"🔗 Extrayendo con EARLY FUSION BioCLIP + DINOv2 de {len(data['observations'])} observaciones...")
        
        # Verificar si tenemos modelo multimodal fine-tuneado
        use_multimodal_finetuned = (self.models['bioclip'].get('finetuned', False) and 
                                   'multimodal_adapter' in self.models['bioclip'] and
                                   'extract_multimodal' in self.models['bioclip'])

        # Limpiar cache CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for obs_id, obs_data in tqdm(data['observations'].items(), desc="Early Fusion Features"):
            all_fused_embeds_obs = []
            all_multimodal_fused_embeds_obs = []
            
            # Crear prompt estructurado
            structured_prompt = self._create_structured_prompt(obs_data)
            obs_data['structured_prompt'] = structured_prompt
            
            # Para cada imagen en la observación
            for img_entry in obs_data.get('image_entries', []):
                fused_embeds_entry = []
                multimodal_fused_embeds_entry = []
                valid_resolutions = 0
                
                for resolution, img_path in img_entry.get('paths', {}).items():
                    if resolution not in self.target_resolutions:
                        continue
                        
                    try:
                        # Cargar imagen
                        img = Image.open(img_path).convert('RGB')
                        
                        # Variables para almacenar embeddings
                        bioclip_embedding = None
                        dinov2_embedding = None
                        bioclip_multimodal_embedding = None
                        
                        # EXTRAER BIOCLIP
                        if 'bioclip' in self.models:
                            try:
                                model_data = self.models['bioclip']
                                model = model_data['model']
                                preprocess = model_data['preprocess']
                                tokenizer = model_data['tokenizer']
                                
                                img_input = preprocess(img).unsqueeze(0).to(self.device)
                                
                                with torch.no_grad():
                                    if use_multimodal_finetuned:
                                        # Usar extracción multimodal fine-tuneada
                                        multimodal_embedding = model_data['extract_multimodal'](img, structured_prompt)
                                        bioclip_multimodal_embedding = multimodal_embedding.squeeze().cpu().numpy()
                                        
                                        # También embedding de imagen pura
                                        image_features = model.encode_image(img_input)
                                        image_features = F.normalize(image_features, p=2, dim=-1)
                                        bioclip_embedding = image_features.squeeze().cpu().numpy()
                                        
                                    else:
                                        # Procesamiento estándar BioCLIP
                                        image_features = model.encode_image(img_input)
                                        image_features = F.normalize(image_features, p=2, dim=-1)
                                        bioclip_embedding = image_features.squeeze().cpu().numpy()
                                        
                                        if structured_prompt:
                                            # Fusión texto-imagen simple
                                            text_tokens = tokenizer([structured_prompt]).to(self.device)
                                            text_features = model.encode_text(text_tokens)
                                            text_features = F.normalize(text_features, p=2, dim=-1)
                                            
                                            img_weight = 1.0 - self.text_image_weight
                                            text_weight = self.text_image_weight
                                            
                                            multimodal_features = img_weight * image_features + text_weight * text_features
                                            multimodal_features = F.normalize(multimodal_features, p=2, dim=-1)
                                            bioclip_multimodal_embedding = multimodal_features.squeeze().cpu().numpy()
                                        else:
                                            bioclip_multimodal_embedding = bioclip_embedding
                            
                            except Exception as e:
                                logger.warning(f"Error con BioCLIP en {img_path}: {e}")
                        
                        # EXTRAER DINOV2
                        if 'dinov2' in self.models:
                            try:
                                model_data = self.models['dinov2']
                                model = model_data['model']
                                preprocess = model_data['preprocess']
                                
                                img_input = preprocess(img).unsqueeze(0).to(self.device)
                                
                                with torch.no_grad():
                                    embedding = model(img_input)
                                    embedding = F.normalize(embedding, p=2, dim=-1)
                                    dinov2_embedding = embedding.squeeze().cpu().numpy()
                            
                            except Exception as e:
                                logger.warning(f"Error con DINOv2 en {img_path}: {e}")
                        
                        if bioclip_embedding is not None and dinov2_embedding is not None:
                            # Concatenar embeddings normalizados
                            fused_embedding = self._concatenate_embeddings(bioclip_embedding, dinov2_embedding)
                            
                            # Aplicar peso de resolución
                            res_weight = self.resolution_weights.get(resolution, 1.0)
                            weighted_fused_embedding = fused_embedding * res_weight
                            
                            fused_embeds_entry.append(weighted_fused_embedding)
                            
                            # También para embeddings multimodales si están disponibles
                            if bioclip_multimodal_embedding is not None:
                                fused_multimodal_embedding = self._concatenate_embeddings(
                                    bioclip_multimodal_embedding, dinov2_embedding
                                )
                                weighted_fused_multimodal_embedding = fused_multimodal_embedding * res_weight
                                multimodal_fused_embeds_entry.append(weighted_fused_multimodal_embedding)
                            
                            valid_resolutions += 1
                            
                        elif bioclip_embedding is not None:
                            # Fallback: solo BioCLIP si DINOv2 falla
                            logger.debug(f"Solo BioCLIP para {img_path} (DINOv2 no disponible)")
                            res_weight = self.resolution_weights.get(resolution, 1.0)
                            weighted_image_embedding = bioclip_embedding * res_weight
                            fused_embeds_entry.append(weighted_image_embedding)
                            
                            if bioclip_multimodal_embedding is not None:
                                weighted_multimodal_embedding = bioclip_multimodal_embedding * res_weight
                                multimodal_fused_embeds_entry.append(weighted_multimodal_embedding)
                            
                            valid_resolutions += 1
                    
                    except Exception as e:
                        logger.warning(f"Error general procesando img {img_path}: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Procesar embeddings fusionados
                if fused_embeds_entry:
                    total_weight = sum(
                        self.resolution_weights.get(res, 1.0) 
                        for res in img_entry.get('paths', {}).keys() 
                        if res in self.target_resolutions and valid_resolutions > 0
                    ) or 1.0
                    
                    if total_weight > 1e-6:
                        all_fused_embeds_obs.append(np.sum(fused_embeds_entry, axis=0) / total_weight)
                
                # Procesar embeddings multimodales fusionados
                if multimodal_fused_embeds_entry:
                    total_weight = sum(
                        self.resolution_weights.get(res, 1.0) 
                        for res in img_entry.get('paths', {}).keys() 
                        if res in self.target_resolutions and valid_resolutions > 0
                    ) or 1.0
                    
                    if total_weight > 1e-6:
                        all_multimodal_fused_embeds_obs.append(np.sum(multimodal_fused_embeds_entry, axis=0) / total_weight)
            
            # Calcular embeddings promedio para la observación
            if all_fused_embeds_obs:
                obs_data['fused_embedding_observation_avg'] = np.mean(all_fused_embeds_obs, axis=0)
                # También mantener compatibilidad con nombres anteriores
                obs_data['image_embedding_observation_avg'] = obs_data['fused_embedding_observation_avg']
            
            if all_multimodal_fused_embeds_obs:
                obs_data['fused_multimodal_embedding_observation_avg'] = np.mean(all_multimodal_fused_embeds_obs, axis=0)
                # También mantener compatibilidad
                obs_data['multimodal_embedding_observation_avg'] = obs_data['fused_multimodal_embedding_observation_avg']
        
        # Limpiar CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Eliminar observaciones sin embeddings
        original_count = len(data['observations'])
        data['observations'] = {o: d for o, d in data['observations'].items() 
                              if 'fused_embedding_observation_avg' in d or 'fused_multimodal_embedding_observation_avg' in d}
        if original_count != len(data['observations']):
            logger.info(f"Eliminadas {original_count - len(data['observations'])} observaciones sin embedding fusionado.")
            
        logger.info("🔗 Extracción con EARLY FUSION completada")
        return data

    def build_index(self, data):
        """Construye índices usando embeddings fusionados"""
        logger.info("🏗️ Construyendo índices con embeddings fusionados...")
        
        # Construir índice para embeddings fusionados (imagen + DINOv2)
        fused_index_data = self._build_single_index(data, 'fused_embedding_observation_avg', 'fused_embeddings')
        
        # Construir índice para embeddings multimodales fusionados si están disponibles
        multimodal_fused_index_data = None
        if self.use_multimodal_processing:
            multimodal_count = sum(1 for obs_data in data['observations'].values() 
                                if 'fused_multimodal_embedding_observation_avg' in obs_data)
            
            if multimodal_count > 0:
                logger.info(f"Construyendo índice multimodal fusionado con {multimodal_count} observaciones...")
                multimodal_fused_index_data = self._build_single_index(data, 'fused_multimodal_embedding_observation_avg', 'multimodal_fused_embeddings')
        
        # Construir prototipos para cada clase
        logger.info("Construyendo prototipos de clase...")
        self._build_class_prototypes(data)
        
        return data, {
            "image": fused_index_data,  # Ahora contiene BioCLIP + DINOv2 fusionados
            "multimodal": multimodal_fused_index_data  # Multimodal + DINOv2 fusionados
        }

    def _build_single_index(self, data, feature_key, index_type):
        """Construye un índice FAISS para un tipo específico de embedding"""
        embeddings, class_indices, observation_ids = [], [], []
        
        for obs_id, obs_data in data['observations'].items():
            if feature_key in obs_data:
                original_class_id = obs_data.get('original_class_id')
                
                if original_class_id is not None and original_class_id in self.class_to_idx:
                    embeddings.append(obs_data[feature_key])
                    class_indices.append(self.class_to_idx[original_class_id])
                    observation_ids.append(obs_id)
        
        if not embeddings:
            logger.warning(f"No embeddings found for {index_type}")
            return None
                
        embeddings_np = np.array(embeddings).astype('float32')
        class_indices_np = np.array(class_indices)
        dim = embeddings_np.shape[1]
        logger.info(f"{index_type} - Embeddings: {embeddings_np.shape}, Dimensión: {dim}")
        
        # Check for NaN/Inf values
        if np.isnan(embeddings_np).any() or np.isinf(embeddings_np).any():
            logger.warning(f"WARNING: NaN or Inf values detected in {index_type}!")
            embeddings_np = np.nan_to_num(embeddings_np)
        
        # Create enhanced HNSW index con parámetros optimizados
        index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.hnsw_ef_construction
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_np)
        
        # Add vectors to index
        try:
            index.add(embeddings_np)
            logger.info(f"{index_type} - Index built with {index.ntotal} vectors.")
        except Exception as e:
            logger.error(f"FAISS {index_type} index build failed: {e}")
            return None
        
        # Configure search parameter con valor optimizado
        index.hnsw.efSearch = self.hnsw_ef_search
        
        return {
            'index': index,
            'embeddings': embeddings_np,
            'class_indices': class_indices_np,
            'observation_ids': observation_ids
        }

    def _build_class_prototypes(self, data):
        """Construye prototipos para embeddings fusionados"""
        data['fused_class_prototypes'] = {}
        data['multimodal_fused_class_prototypes'] = {}
        
        # Prototipos para embeddings fusionados
        self._build_prototype_by_type(data, 'fused_embedding_observation_avg', 'fused_class_prototypes')
        
        # Prototipos para embeddings multimodales fusionados
        if self.use_multimodal_processing:
            multimodal_count = sum(1 for obs_data in data['observations'].values() 
                                if 'fused_multimodal_embedding_observation_avg' in obs_data)
            
            if multimodal_count > 0:
                self._build_prototype_by_type(data, 'fused_multimodal_embedding_observation_avg', 'multimodal_fused_class_prototypes')
        
        # Mantener compatibilidad con nombres anteriores
        if 'fused_class_prototypes' in data:
            data['image_class_prototypes'] = data['fused_class_prototypes']
        if 'multimodal_fused_class_prototypes' in data:
            data['multimodal_class_prototypes'] = data['multimodal_fused_class_prototypes']

    def _build_prototype_by_type(self, data, feature_key, prototype_key):
        """Construye prototipos para un tipo específico de embedding"""
        class_embeddings = {}
        
        for obs_id, obs_data in data['observations'].items():
            if feature_key not in obs_data:
                continue
                
            class_id = obs_data.get('original_class_id', -1)
            if class_id == -1 or class_id not in self.class_to_idx:
                continue
                
            class_idx = self.class_to_idx[class_id]
            
            if class_idx not in class_embeddings:
                class_embeddings[class_idx] = []
                
            class_embeddings[class_idx].append(obs_data[feature_key])
        
        unique_classes = set(class_embeddings.keys())
        
        for class_idx in unique_classes:
            embeddings = class_embeddings[class_idx]
            
            if embeddings:
                # Calcular centroide
                centroid = np.mean(embeddings, axis=0)
                
                # Normalizar centroide
                norm = np.linalg.norm(centroid)
                if norm > 1e-6:
                    data[prototype_key][class_idx] = (centroid / norm).astype('float32')
        
        logger.info(f"Prototipos construidos para {len(data[prototype_key])} clases en {prototype_key}")

    def _predict_multimodal(self, obs_data, train_data, index_data, top_k=10):
        """Predicción simplificada usando embeddings fusionados"""
        
        # Determinar qué tipo de embedding usar
        has_multimodal_fused = ('fused_multimodal_embedding_observation_avg' in obs_data and 
                               index_data.get('multimodal') is not None)
        
        if has_multimodal_fused:
            logger.debug("Usando embeddings multimodales fusionados (BioCLIP+texto+DINOv2)")
            predictions = self._predict_by_type(
                obs_data, 'fused_multimodal_embedding_observation_avg',
                train_data, index_data['multimodal'], 
                'multimodal_fused_class_prototypes', top_k
            )
        else:
            logger.debug("Usando embeddings de imagen fusionados (BioCLIP+DINOv2)")
            predictions = self._predict_by_type(
                obs_data, 'fused_embedding_observation_avg',
                train_data, index_data['image'], 
                'fused_class_prototypes', top_k
            )
        
        return predictions

    def _predict_by_type(self, obs_data, feature_key, train_data, index_data, prototype_key, top_k=10):
        """Método core de predicción para embeddings fusionados"""
        # Obtener embedding de la query
        query_embed = obs_data.get(feature_key)
        if query_embed is None:
            logger.warning(f"No {feature_key} available for prediction")
            return []
        
        # Normalizar embedding de query
        query_norm = np.linalg.norm(query_embed)
        if query_norm < 1e-6:
            logger.warning(f"Query embedding has near-zero norm for {feature_key}")
            return []
        
        query_embed_norm = (query_embed / query_norm).astype('float32').reshape(1, -1)
        
        # Verificar índice válido
        if index_data is None or 'index' not in index_data or index_data['index'] is None:
            logger.warning(f"Index data is invalid for {feature_key}")
            return []
        
        if index_data['index'].ntotal == 0:
            logger.warning(f"FAISS index is empty for {feature_key}")
            return []
        
        # Realizar búsqueda k-NN con parámetros optimizados
        k_search = min(self.k_neighbors * self.k_search_multiplier, index_data['index'].ntotal)
        try:
            similarities_hnsw, indices_hnsw = index_data['index'].search(query_embed_norm, k_search)
        except Exception as e:
            logger.error(f"FAISS search failed for {feature_key}: {e}")
            return []
        
        # Verificar resultados de búsqueda
        if indices_hnsw.size == 0 or similarities_hnsw.size == 0:
            logger.warning(f"FAISS search returned empty results for {feature_key}")
            return []
        
        # Inicializar scores con múltiples estrategias
        knn_scores = {}
        centroid_scores = {}
        medoid_scores = {}
        metadata_scores = {}
        
        # 1. Procesamiento kNN básico
        for i in range(min(self.k_neighbors, len(indices_hnsw[0]))):
            idx = indices_hnsw[0][i]
            sim = similarities_hnsw[0][i]
            
            # Verificar índice válido
            if idx < 0 or idx >= len(index_data['class_indices']) or sim < 0:
                continue
            
            class_idx = index_data['class_indices'][idx]
            observation_id = index_data['observation_ids'][idx]
            
            # Acumular score para esta clase
            if class_idx not in knn_scores:
                knn_scores[class_idx] = 0.0
            
            # Peso por posición
            rank_weight = 1.0 / np.log(2 + i)
            knn_scores[class_idx] += sim * rank_weight
            
            # 2. Procesamiento de metadatos
            if 'metadata' in obs_data:
                query_metadata = obs_data['metadata']
                train_obs = train_data['observations'].get(observation_id)
                
                if train_obs and 'metadata' in train_obs:
                    train_metadata = train_obs['metadata']
                    metadata_sim = 0.0
                    
                    # Habitat
                    if query_metadata.get('habitat') and train_metadata.get('habitat'):
                        if query_metadata['habitat'].lower() == train_metadata['habitat'].lower():
                            metadata_sim += self.metadata_sim_weights.get('habitat', 0.5)
                    
                    # Substrate
                    if query_metadata.get('substrate') and train_metadata.get('substrate'):
                        if query_metadata['substrate'].lower() == train_metadata['substrate'].lower():
                            metadata_sim += self.metadata_sim_weights.get('substrate', 0.4)
                    
                    # Bioregion
                    if query_metadata.get('biogeographicalRegion') and train_metadata.get('biogeographicalRegion'):
                        if query_metadata['biogeographicalRegion'].lower() == train_metadata['biogeographicalRegion'].lower():
                            metadata_sim += self.metadata_sim_weights.get('bioregion', 0.2)
                    
                    # Month
                    if query_metadata.get('month', -1) != -1 and train_metadata.get('month', -1) != -1:
                        month_q = query_metadata['month']
                        month_t = train_metadata['month']
                        month_diff = min(abs(month_q - month_t), 12 - abs(month_q - month_t))
                        if month_diff <= 1:
                            metadata_sim += self.metadata_sim_weights.get('month', 0.3)
                        elif month_diff <= 2:
                            metadata_sim += self.metadata_sim_weights.get('month', 0.3) * 0.5
                    
                    if metadata_sim > 0:
                        if class_idx not in metadata_scores:
                            metadata_scores[class_idx] = 0.0
                        metadata_scores[class_idx] += metadata_sim * rank_weight
        
        # 3. Procesar similitud con centroides/prototipos
        if prototype_key in train_data:
            for class_idx, prototype in train_data[prototype_key].items():
                centroid_sim = np.dot(query_embed_norm.reshape(-1), prototype)
                if centroid_sim > 0:
                    centroid_scores[class_idx] = float(centroid_sim)
        
        # 4. Procesar similitud con medoides
        if 'class_medoids' in train_data:
            for class_idx, medoid_obs_id in train_data['class_medoids'].items():
                medoid_obs = train_data['observations'].get(medoid_obs_id)
                if medoid_obs and feature_key in medoid_obs:
                    medoid_embed = medoid_obs[feature_key]
                    medoid_norm = np.linalg.norm(medoid_embed)
                    if medoid_norm > 1e-6:
                        medoid_embed_norm = medoid_embed / medoid_norm
                        medoid_sim = np.dot(query_embed_norm.reshape(-1), medoid_embed_norm)
                        if medoid_sim > 0:
                            medoid_scores[class_idx] = float(medoid_sim)
        
        # Combinar scores de todas las estrategias con pesos optimizados
        strategies = [
            ('knn', knn_scores, self.ensemble_weights.get('knn', 0.5)),
            ('centroid', centroid_scores, self.ensemble_weights.get('centroid', 0.3)),
            ('medoid', medoid_scores, self.ensemble_weights.get('medoid', 0.2)),
            ('metadata', metadata_scores, self.ensemble_weights.get('metadata', 0.45))
        ]
        
        combined_scores = {}
        
        for strategy_name, scores, weight in strategies:
            if not scores:
                continue
                
            # Normalizar scores para esta estrategia
            max_score = max(scores.values()) if scores else 0
            if max_score > 0:
                normalized_scores = {cls: score / max_score for cls, score in scores.items()}
                
                # Añadir a scores combinados
                for cls, score in normalized_scores.items():
                    if cls not in combined_scores:
                        combined_scores[cls] = 0.0
                    combined_scores[cls] += score * weight
        
        # Aplicar boost para especies raras con parámetros optimizados
        if self.use_rare_species_boost and hasattr(self, 'class_frequency') and self.class_frequency:
            for cls_idx in combined_scores:
                original_class_id = self.idx_to_class.get(cls_idx)
                if original_class_id in self.class_frequency:
                    class_freq = self.class_frequency[original_class_id]
                    if class_freq <= self.rare_species_threshold:
                        boost_factor = 1.0 + (self.rare_species_boost_factor * 
                                            (1.0 - class_freq / self.rare_species_threshold))
                        combined_scores[cls_idx] *= boost_factor
        
        # Ordenar por score final
        sorted_scores = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Mapear índices internos a IDs originales de clase
        results = []
        for class_idx, score in sorted_scores[:top_k]:
            original_class_id = self.idx_to_class.get(class_idx)
            if original_class_id is not None:
                results.append((original_class_id, score))
        
        return results

    # ===== FUNCIONES AUXILIARES =====
    
    def _build_ecological_context_model(self, train_data):
        """Construye modelos de probabilidad para especies basados en contexto ecológico"""
        logger.info("Construyendo modelo contextual ecológico...")
        
        # Inicializar contadores
        for obs_id, obs_data in train_data['observations'].items():
            species_id = obs_data.get('original_class_id', -1)
            if species_id == -1:
                continue
                
            if 'metadata' in obs_data:
                metadata = obs_data['metadata']
                
                # Procesar hábitat
                habitat = metadata.get('habitat', '').lower().strip()
                if habitat:
                    self.habitat_species[habitat][species_id] += 1
                    self.habitat_counts[habitat] += 1
                
                # Procesar sustrato
                substrate = metadata.get('substrate', '').lower().strip()
                if substrate:
                    self.substrate_species[substrate][species_id] += 1
                    self.substrate_counts[substrate] += 1
                
                # Procesar región biogeográfica
                bioregion = metadata.get('biogeographicalRegion', '').lower().strip()
                if bioregion:
                    self.region_species[bioregion][species_id] += 1
                    self.region_counts[bioregion] += 1
                
                # Procesar mes
                month = metadata.get('month', -1)
                if month != -1 and 1 <= month <= 12:
                    self.month_species[month][species_id] += 1
                    self.month_counts[month] += 1
        
        # Calcular probabilidades con suavizado Laplace
        alpha = 0.1
        num_classes = len(self.class_to_idx)
        
        for habitat, species_counts in self.habitat_species.items():
            total = self.habitat_counts[habitat]
            for species_id in self.class_to_idx.keys():
                count = species_counts[species_id]
                self.habitat_probs[habitat][species_id] = (count + alpha) / (total + alpha * num_classes)
        
        for substrate, species_counts in self.substrate_species.items():
            total = self.substrate_counts[substrate]
            for species_id in self.class_to_idx.keys():
                count = species_counts[species_id]
                self.substrate_probs[substrate][species_id] = (count + alpha) / (total + alpha * num_classes)
        
        for region, species_counts in self.region_species.items():
            total = self.region_counts[region]
            for species_id in self.class_to_idx.keys():
                count = species_counts[species_id]
                self.region_probs[region][species_id] = (count + alpha) / (total + alpha * num_classes)
        
        for month, species_counts in self.month_species.items():
            total = self.month_counts[month]
            for species_id in self.class_to_idx.keys():
                count = species_counts[species_id]
                self.month_probs[month][species_id] = (count + alpha) / (total + alpha * num_classes)
        
        logger.info(f"Modelo contextual construido - Hábitats: {len(self.habitat_species)}, Sustratos: {len(self.substrate_species)}")

    def _apply_ecological_context(self, base_predictions, obs_data, top_k=10):
        """Aplica contexto ecológico para reordenar predicciones"""
        if not base_predictions or not hasattr(self, 'habitat_probs'):
            return base_predictions[:top_k]
        
        # Extraer contexto de la observación
        context = {}
        if 'metadata' in obs_data:
            metadata = obs_data['metadata']
            
            habitat = str(metadata.get('habitat', '')).lower().strip()
            if habitat:
                context['habitat'] = habitat
                
            substrate = str(metadata.get('substrate', '')).lower().strip()
            if substrate:
                context['substrate'] = substrate
                
            bioregion = str(metadata.get('biogeographicalRegion', '')).lower().strip()
            if bioregion:
                context['region'] = bioregion
                
            month = metadata.get('month', -1)
            if 1 <= month <= 12:
                context['month'] = month
        
        if not context:
            return base_predictions[:top_k]
        
        # Aplicar factores de contexto con pesos optimizados
        final_scores = {}
        
        for species_id, base_score in base_predictions:
            final_score = base_score
            context_boost = 1.0
            
            # Habitat
            if 'habitat' in context and context['habitat'] in self.habitat_probs:
                habitat_prob = self.habitat_probs[context['habitat']].get(species_id, 0.0)
                if habitat_prob > 0:
                    habitat_boost = 1.0 + self.context_weights['habitat'] * (1.0 + np.log10(max(habitat_prob * 10, 1.0)))
                    context_boost *= habitat_boost
            
            # Substrate
            if 'substrate' in context and context['substrate'] in self.substrate_probs:
                substrate_prob = self.substrate_probs[context['substrate']].get(species_id, 0.0)
                if substrate_prob > 0:
                    substrate_boost = 1.0 + self.context_weights['substrate'] * (1.0 + np.log10(max(substrate_prob * 10, 1.0)))
                    context_boost *= substrate_boost
            
            # Region
            if 'region' in context and context['region'] in self.region_probs:
                region_prob = self.region_probs[context['region']].get(species_id, 0.0)
                if region_prob > 0:
                    region_boost = 1.0 + self.context_weights['region'] * (1.0 + np.log10(max(region_prob * 10, 1.0)))
                    context_boost *= region_boost
            
            # Month
            if 'month' in context and context['month'] in self.month_probs:
                month_prob = self.month_probs[context['month']].get(species_id, 0.0)
                if month_prob > 0:
                    month_boost = 1.0 + self.context_weights['month'] * (1.0 + np.log10(max(month_prob * 10, 1.0)))
                    context_boost *= month_boost
            
            final_score *= context_boost
            final_scores[species_id] = final_score
        
        ranked_predictions = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_predictions[:top_k]

    def _build_taxonomic_hierarchy(self, train_data):
        """Construye estructura jerárquica taxonómica"""
        self.species_to_taxonomy = {}
        self.taxonomy_to_species = {'genus': {}, 'family': {}, 'order': {}, 'class': {}, 'phylum': {}}
        
        for obs_id, obs_data in train_data['observations'].items():
            species_id = obs_data.get('original_class_id', -1)
            if species_id == -1 or 'metadata' not in obs_data:
                continue
                
            metadata = obs_data['metadata']
            
            if species_id not in self.species_to_taxonomy:
                self.species_to_taxonomy[species_id] = {}
            
            for tax_level in ['genus', 'family', 'order', 'class', 'phylum']:
                if tax_level in metadata and metadata[tax_level]:
                    tax_name = str(metadata[tax_level]).lower().strip()
                    if tax_name and tax_name != 'nan':
                        self.species_to_taxonomy[species_id][tax_level] = tax_name
                        
                        if tax_name not in self.taxonomy_to_species[tax_level]:
                            self.taxonomy_to_species[tax_level][tax_name] = []
                        
                        if species_id not in self.taxonomy_to_species[tax_level][tax_name]:
                            self.taxonomy_to_species[tax_level][tax_name].append(species_id)
        
        logger.info(f"Construida jerarquía taxonómica con {len(self.species_to_taxonomy)} especies")

    def _hierarchical_predict(self, base_predictions, obs_data, top_k=10):
        """Aplica clasificación jerárquica basada en taxonomía"""
        if not base_predictions:
            return []
        
        # Extraer taxonomía de la observación
        query_taxonomy = {}
        if 'metadata' in obs_data:
            metadata = obs_data['metadata']
            for tax_level in ['genus', 'family', 'order', 'class', 'phylum']:
                if tax_level in metadata and metadata[tax_level]:
                    tax_name = str(metadata[tax_level]).lower().strip()
                    if tax_name and tax_name != 'nan':
                        query_taxonomy[tax_level] = tax_name
        
        # Análisis taxonómico de predicciones base
        taxa_votes = {level: Counter() for level in ['genus', 'family', 'order', 'class', 'phylum']}
        
        for species_id, score in base_predictions:
            if species_id in self.species_to_taxonomy:
                species_taxonomy = self.species_to_taxonomy[species_id]
                for level, taxon in species_taxonomy.items():
                    taxa_votes[level][taxon] += score
        
        # Normalizar votos
        normalized_taxa_votes = {}
        for level, votes in taxa_votes.items():
            if votes:
                max_vote = max(votes.values())
                if max_vote > 0:
                    normalized_taxa_votes[level] = {taxon: vote/max_vote for taxon, vote in votes.items()}
        
        # Re-ranking usando información taxonómica
        final_scores = {}
        
        for species_id, base_score in base_predictions:
            final_score = base_score
            
            if species_id in self.species_to_taxonomy:
                species_taxonomy = self.species_to_taxonomy[species_id]
                
                for level, taxon in species_taxonomy.items():
                    if level in query_taxonomy and taxon == query_taxonomy[level]:
                        tax_bonus = self.taxonomy_level_weights[level] * 2.0
                        final_score *= (1.0 + tax_bonus)
                    elif level in normalized_taxa_votes and taxon in normalized_taxa_votes[level]:
                        consensus_strength = normalized_taxa_votes[level][taxon]
                        tax_bonus = self.taxonomy_level_weights[level] * consensus_strength
                        final_score *= (1.0 + tax_bonus)
            
            final_scores[species_id] = final_score
        
        ranked_predictions = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_predictions[:top_k]

    def _build_enhanced_prototypes(self, train_data):
        """Crea medoides para cada clase"""
        logger.info("Construyendo medoides de clase...")
        
        class_embeddings = {}
        for obs_id, obs_data in train_data['observations'].items():
            species_id = obs_data.get('original_class_id', -1)
            if species_id == -1 or 'fused_embedding_observation_avg' not in obs_data:
                continue
                
            if species_id not in class_embeddings:
                class_embeddings[species_id] = []
            
            class_embeddings[species_id].append({
                'obs_id': obs_id,
                'embedding': obs_data['fused_embedding_observation_avg']
            })
        
        train_data['class_medoids'] = {}
        
        for species_id, samples in class_embeddings.items():
            if len(samples) == 0:
                continue
                
            if len(samples) == 1:
                train_data['class_medoids'][species_id] = samples[0]['obs_id']
                continue
            
            embeddings = np.array([s['embedding'] for s in samples])
            obs_ids = [s['obs_id'] for s in samples]
            
            # Calcular centroide
            centroid = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-6:
                centroid = centroid / norm
            
            # Encontrar medoide (muestra más cercana al centroide)
            distances_to_centroid = [cosine(embeddings[i], centroid) for i in range(len(embeddings))]
            medoid_idx = np.argmin(distances_to_centroid)
            train_data['class_medoids'][species_id] = obs_ids[medoid_idx]
        
        logger.info(f"Medoides identificados para {len(train_data['class_medoids'])} clases")
        return train_data

    # ===== FINE-TUNING MULTIMODAL (OPCIONAL) =====
    
    class FungiMultimodalDataset(Dataset):
        def __init__(self, data_cache, transform=None, class_to_idx=None, pipeline=None):
            self.obs_ids = []
            self.images_paths, self.prompts, self.labels = [], [], []
            self.transform = transform
            self.class_to_idx = class_to_idx
            self.pipeline = pipeline

            for obs_id, obs_data in data_cache['observations'].items():
                class_id = obs_data.get('original_class_id', -1)
                if class_id == -1:
                    continue
                
                label_idx = self.class_to_idx.get(class_id, -1)
                if label_idx == -1:
                    continue
                
                prompt = self.pipeline._create_structured_prompt(obs_data)
                
                best_resolution_found = False
                for img_entry in obs_data.get('image_entries', []):
                    for res in ['fullsize', '720p', '500p']:
                        if res in img_entry.get('paths', {}):
                            path = img_entry['paths'][res]
                            self.obs_ids.append(obs_id)
                            self.images_paths.append(path)
                            self.labels.append(label_idx)
                            self.prompts.append(prompt)
                            best_resolution_found = True
                            break
                    if best_resolution_found:
                        break

        def __len__(self):
            return len(self.images_paths)

        def __getitem__(self, idx):
            img_path = self.images_paths[idx]
            prompt = self.prompts[idx] 
            label = self.labels[idx]
            
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, prompt, label
            except Exception:
                return torch.zeros((3, 224, 224)), prompt, label

    def finetune_bioclip_with_early_fusion(self, train_data, val_data,
                                          epochs=10,              # Según memoria
                                          batch_size=32,          # Según memoria
                                          lr_model=3e-5,          # Según memoria
                                          lr_adapter=2e-4,        # Según memoria
                                          beta=0.75):             # Según memoria
        """
        Fine-tuning de BioCLIP adaptado para early fusion con parámetros optimizados
        """

        class OptimizedFocalLoss(nn.Module):
            def __init__(self, alpha=0.3, gamma=1.5, label_smoothing=0.1):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.label_smoothing = label_smoothing
                self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')
            
            def forward(self, inputs, targets):
                ce_loss = self.ce_loss(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()

        # Transformaciones
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),       
            transforms.RandomHorizontalFlip(p=0.5),                    
            transforms.RandomVerticalFlip(p=0.5),                      
            transforms.RandomRotation(degrees=20),                     
            transforms.ColorJitter(brightness=0.2, contrast=0.2),      
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                std=(0.26862954, 0.26130258, 0.27577711))
        ])

        # Datasets
        train_ds = self.FungiMultimodalDataset(train_data, transform=train_transform, 
                                              class_to_idx=self.class_to_idx, pipeline=self)
        val_ds = self.FungiMultimodalDataset(val_data, transform=self.models['bioclip']['preprocess'], 
                                            class_to_idx=self.class_to_idx, pipeline=self)

        if len(train_ds) == 0:
            logger.error("Dataset vacío")
            return None, 0.0

        # Weighted sampler con parámetros optimizados
        class_counts = Counter(train_ds.labels)
        max_count = max(class_counts.values())
        class_weights = [max(0.03, (max_count / class_counts[label]) ** (1/3)) for label in train_ds.labels]
        weights = torch.DoubleTensor(class_weights)
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

        # DataLoaders con batch_size optimizado
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, 
                                 num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                               num_workers=0, pin_memory=False)

        logger.info(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}")

        # Modelo BioCLIP
        model = self.models['bioclip']['model'].to(self.device)
        tokenizer = self.models['bioclip']['tokenizer']

        # Congelar modelo base excepto últimas capas
        for param in model.parameters():
            param.requires_grad = False

        # Descongelar las 6 ÚLTIMAS CAPAS COMPLETAS del transformer visual
        if hasattr(model, 'visual') and hasattr(model.visual, 'transformer') and hasattr(model.visual.transformer, 'resblocks'):
            total_layers = len(model.visual.transformer.resblocks)
            layers_to_unfreeze = 6  # 6 últimas capas según memoria
            
            logger.info(f"Total de capas en ViT: {total_layers}")
            logger.info(f"Descongelando las últimas {layers_to_unfreeze} capas")
            
            for i in range(total_layers - layers_to_unfreeze, total_layers):
                logger.info(f"Descongelando capa {i}")
                for param in model.visual.transformer.resblocks[i].parameters():
                    param.requires_grad = True

        # Descongelar capas de normalización y proyección
        for name, param in model.named_parameters():
            if 'ln_' in name or 'norm' in name or 'visual.proj' in name:
                param.requires_grad = True
                logger.info(f"Descongelando: {name}")

        num_classes = len(self.class_to_idx)
        
        # Adapter diseñado para embeddings fusionados con arquitectura optimizada según memoria
        expected_fused_dim = model.visual.output_dim  # Solo BioCLIP por ahora
        
        multimodal_adapter = nn.Sequential(
            nn.Linear(expected_fused_dim, 2048),  
            nn.LayerNorm(2048),                        
            nn.GELU(),                                 
            nn.Dropout(0.3),                          
            nn.Linear(2048, 1024),                    
            nn.LayerNorm(1024),                       
            nn.GELU(),                                
            nn.Dropout(0.4),                          
            nn.Linear(1024, num_classes)              
        ).to(self.device)

        logger.info(f"Adapter creado para dim {expected_fused_dim} -> {num_classes} clases")

        # Optimizador con LR optimizados según memoria
        model_params = [p for p in model.parameters() if p.requires_grad]
        adapter_params = list(multimodal_adapter.parameters())
        
        if len(model_params) == 0:
            optimizer = optim.AdamW(adapter_params, lr=lr_adapter, weight_decay=0.01)
        else:
            optimizer = optim.AdamW([
                {'params': model_params, 'lr': lr_model, 'weight_decay': 1e-6},
                {'params': adapter_params, 'lr': lr_adapter, 'weight_decay': 5e-5}
            ])
            logger.info(f"Optimizador con LRs optimizados - Modelo: {lr_model}, Adapter: {lr_adapter}")

        # Scheduler con parámetros optimizados
        warmup_steps = min(300, len(train_loader) * 2)
        total_steps = epochs * len(train_loader)
        
        def lr_lambda_cosine_warmup(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_cosine_warmup)

        # Loss function optimizada según memoria
        criterion = OptimizedFocalLoss(alpha=0.3, gamma=1.5, label_smoothing=0.1)

        logger.info("Iniciando fine-tuning para EARLY FUSION con parámetros optimizados...")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Beta: {beta}")

        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(epochs):
            # Training
            model.train()
            multimodal_adapter.train()
            
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")
            for images, prompts, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                try:
                    # Extraer features de imagen
                    image_features = model.encode_image(images)
                    
                    # Extraer features de texto
                    text_tokens = tokenizer(prompts).to(self.device)
                    text_features = model.encode_text(text_tokens)
                    
                    # Fusión multimodal (como preparación para early fusion completa)
                    fused_features = beta * image_features + (1 - beta) * text_features
                    
                    # Clasificación
                    outputs = multimodal_adapter(fused_features)
                    loss = criterion(outputs, labels)
                    
                    # Backward
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                    torch.nn.utils.clip_grad_norm_(multimodal_adapter.parameters(), 0.3)
                    optimizer.step()
                    scheduler.step()
                    
                    # Stats
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*correct/total:.2f}%'
                    })
                    
                except Exception as e:
                    logger.warning(f"Error en batch: {e}")
                    continue
            
            train_acc = correct / total if total > 0 else 0.0
            avg_loss = running_loss / len(train_loader)
            
            # Validation
            model.eval()
            multimodal_adapter.eval()
            
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation")
                for images, prompts, labels in pbar_val:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    try:
                        image_features = model.encode_image(images)
                        text_tokens = tokenizer(prompts).to(self.device)
                        text_features = model.encode_text(text_tokens)
                        fused_features = beta * image_features + (1 - beta) * text_features
                        outputs = multimodal_adapter(fused_features)
                        
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                        pbar_val.set_postfix({'Acc': f'{100.*val_correct/val_total:.2f}%'})
                        
                    except Exception as e:
                        continue
            
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            
            logger.info(f"Epoch {epoch+1}/{epochs}:")
            logger.info(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"  Val Acc: {val_acc:.4f}")
            
            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    'adapter_state_dict': multimodal_adapter.state_dict(),
                    'val_acc': val_acc,
                    'beta': beta,
                    'num_classes': num_classes
                }
                logger.info(f"  ✓ Nuevo mejor modelo guardado (Val Acc: {val_acc:.4f})")

        # Función para extraer embeddings multimodales
        def extract_multimodal_embeddings(img, prompt="A photograph of fungi species"):
            img_tensor = self.models['bioclip']['preprocess'](img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = model.encode_image(img_tensor)
                text_tokens = tokenizer([prompt]).to(self.device)
                text_features = model.encode_text(text_tokens)
                fused_features = beta * image_features + (1 - beta) * text_features
                return fused_features.cpu().numpy()

        # Actualizar el modelo en el pipeline
        self.models['bioclip']['model'] = model
        self.models['bioclip']['finetuned'] = True
        self.models['bioclip']['multimodal_adapter'] = multimodal_adapter
        self.models['bioclip']['extract_multimodal'] = extract_multimodal_embeddings
        self.models['bioclip']['beta'] = beta

        # Guardar el modelo
        models_dir = Path(self.base_path) / "models"
        models_dir.mkdir(exist_ok=True, parents=True)
        
        if best_model_state:
            model_path = models_dir / "bioclip_early_fusion_finetuned.pt"
            torch.save(best_model_state, model_path)
            logger.info(f"✅ Modelo early fusion guardado en: {model_path}")
            logger.info(f"🎯 Mejor Val Acc: {best_val_acc:.4f}")

        return model, best_val_acc

    # ===== FUNCIONES PRINCIPALES =====
    
    def prepare_data_and_index(self):
        """Prepara datos e índice con early fusion y parámetros optimizados"""
        logger.info("🔗 Preparando pipeline con EARLY FUSION BioCLIP + DINOv2 y parámetros optimizados...")
        
        # Cargar datos
        logger.info("Cargando datos de train...")
        train_data = self.load_data(split='train')
        if not train_data or not train_data.get('observations'):
            raise RuntimeError("Fallo al cargar datos de train.")
        
        logger.info("Cargando datos de val...")
        val_data = self.load_data(split='val')
        if not val_data or not val_data.get('observations'):
            logger.warning("No se cargaron datos de validación.")
            val_data = {'observations': {}}
        
        # Extracción con early fusion
        logger.info("🔗 Extrayendo features con EARLY FUSION...")
        train_data = self.extract_features_multimodal(train_data)
        
        if val_data.get('observations'):
            val_data = self.extract_features_multimodal(val_data)
        
        # Construir índices
        logger.info("🏗️ Construyendo índices con embeddings fusionados...")
        train_data, index_data = self.build_index(train_data)
        self.train_data_cache = train_data
        self.index_data_cache = index_data
        self.val_data_cache = val_data
        
        # Construir modelos auxiliares
        logger.info("Construyendo modelos auxiliares...")
        self._build_taxonomic_hierarchy(train_data)
        self._build_ecological_context_model(train_data)
        self._build_enhanced_prototypes(train_data)
            
        logger.info("✅ Pipeline con EARLY FUSION y parámetros optimizados listo")
        return self.train_data_cache, self.val_data_cache, self.index_data_cache

    def predict_observation(self, obs_data, train_data, index_data, top_k=10):
        """Predice para una observación individual"""
        try:
            # Usar predicción con embeddings fusionados
            predictions = self._predict_multimodal(obs_data, train_data, index_data, top_k)
            
            if not predictions:
                logger.warning("No predictions generated, using fallback")
                fallback_classes = list(self.class_to_idx.keys())[:top_k]
                return [(cls_id, 0.1) for cls_id in fallback_classes]
            
            # Aplicar contexto ecológico
            if hasattr(self, 'habitat_probs') and len(self.habitat_probs) > 0:
                try:
                    predictions = self._apply_ecological_context(predictions, obs_data, top_k)
                except Exception as eco_e:
                    logger.warning(f"Error aplicando contexto ecológico: {eco_e}")
            
            # Aplicar clasificación jerárquica
            if hasattr(self, 'species_to_taxonomy') and len(self.species_to_taxonomy) > 0:
                try:
                    predictions = self._hierarchical_predict(predictions, obs_data, top_k)
                except Exception as hier_e:
                    logger.warning(f"Error aplicando clasificación jerárquica: {hier_e}")
            
            # Asegurar top_k predicciones
            final_predictions = predictions[:top_k]
            
            if len(final_predictions) < top_k:
                remaining_classes = [cls_id for cls_id in self.class_to_idx.keys() 
                                   if cls_id not in [p[0] for p in final_predictions]]
                
                for cls_id in remaining_classes[:top_k - len(final_predictions)]:
                    final_predictions.append((cls_id, 0.01))
            
            return final_predictions[:top_k]
            
        except Exception as e:
            logger.error(f"Error crítico en predict_observation: {e}")
            fallback_classes = list(self.class_to_idx.keys())[:top_k]
            return [(cls_id, 0.01) for cls_id in fallback_classes]

    def evaluate(self, train_data, val_data, index_data, k=5):
        """Evalúa el rendimiento del modelo en datos de validación"""
        if not val_data or not val_data.get('observations'):
            logger.warning("No hay datos de validación.")
            return 0.0
        
        logger.info(f"Evaluando R@{k} en {len(val_data['observations'])} observaciones...")
        
        if not index_data or 'image' not in index_data or index_data['image'] is None:
            logger.error("Invalid index data")
            return 0.0
        
        if index_data['image']['index'].ntotal == 0:
            logger.error("Empty FAISS index")
            return 0.0
        
        if not self.class_to_idx or not self.idx_to_class:
            logger.error("Class mapping is empty")
            return 0.0
        
        correct, total = 0, 0
        
        for obs_id, obs_data in tqdm(val_data['observations'].items(), desc="Evaluando Validación"):
            true_orig_cid = obs_data.get('original_class_id')
            
            if true_orig_cid is None or true_orig_cid not in self.class_to_idx:
                continue
            
            preds_scores = self.predict_observation(obs_data, train_data, index_data, top_k=k)
            pred_orig_cids = [p[0] for p in preds_scores]
            
            if true_orig_cid in pred_orig_cids:
                correct += 1
                    
            total += 1
            
        if total == 0:
            logger.warning("No valid observations were evaluated")
            return 0.0
            
        recall = correct / total
        logger.info(f"R@{k} (Val): {recall:.4f} ({correct}/{total})")
        
        return recall

    def predict_test(self, train_data, index_data, output_file='submission.csv', k=10):
        """Genera predicciones para el dataset de test"""
        logger.info("--- Predicción de Test con EARLY FUSION ---")
        
        test_data = self.load_data(split='test')
        if not test_data or not test_data.get('observations'):
            logger.warning("No se pudieron cargar datos de test.")
            return None
        
        logger.info("🔗 Extrayendo features fusionados para test...")
        test_data = self.extract_features_multimodal(test_data)
        
        logger.info(f"Generando predicciones Top-{k} para {len(test_data['observations'])} observaciones...")
        results = []
        
        for obs_id, obs_data in tqdm(test_data['observations'].items(), desc="Prediciendo Test"):
            preds_scores = self.predict_observation(obs_data, train_data, index_data, top_k=k)
            class_ids = [p[0] for p in preds_scores]
            
            while len(class_ids) < k:
                class_ids.append(-1)
            
            results.append({
                'observationId': obs_id, 
                'predictions': ' '.join(map(str, class_ids[:k]))
            })
        
        results_df = pd.DataFrame(results)
        output_path = self.base_path / output_file
        
        results_df.to_csv(output_path, index=False)
        logger.info(f"Resultados guardados en: {output_path}")
        
        return results_df