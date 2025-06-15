"""
2. Procesamiento de Datos
Pipeline Multimodal FungiCLEF 2025

Este archivo contiene:
- Carga de datos desde CSV siguiendo la estructura del repositorio
- Extracción de características multimodales
- Creación de prompts estructurados
- Procesamiento de metadatos y captions
"""

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
        
        # Convert category_id to numeric safely
        metadata_df['category_id'] = pd.to_numeric(metadata_df['category_id'], errors='coerce').fillna(-1).astype(int)
        
        # Build class mapping from training data
        if split == 'train':
            logger.info("Building class mapping from training data...")
            
            # Get unique classes
            valid_category_mask = metadata_df['category_id'] != -1
            unique_classes = sorted(metadata_df[valid_category_mask]['category_id'].unique())
            
            # Build class mappings
            self.class_to_idx.clear()
            self.idx_to_class.clear()
            self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            self.num_classes = len(self.class_to_idx)
            
            # Calculate class frequencies
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
    
    # Lista de columnas taxonómicas que debemos preservar
    taxonomic_cols = ['genus', 'family', 'order', 'class', 'phylum', 'species']
    
    # Lista de metadatos importantes
    metadata_cols = ['habitat', 'substrate', 'biogeographicalRegion', 'month']
    
    # Procesar filas
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc=f"Processing {split}"):
        try:
            obs_id = row['observationID']
            img_filename = row['filename']
            
            # For test, category_id may not exist or be NaN. Default to -1.
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
                # Inicializar metadata
                metadata = {}
                
                # Procesar campos de metadatos
                for col in metadata_cols:
                    if col in row and pd.notna(row[col]):
                        # Convertir tipos según el campo
                        if col in ['month']:
                            metadata[col] = int(row[col]) if pd.notna(row.get(col)) else -1
                        else:
                            metadata[col] = str(row.get(col, '')).strip()
                
                # Procesar campos taxonómicos
                for col in taxonomic_cols:
                    if col in row and pd.notna(row[col]):
                        tax_value = str(row[col]).strip()
                        if tax_value and tax_value != 'nan':  # Solo guardar si no está vacío
                            metadata[col] = tax_value
                
                data['observations'][obs_id]['metadata'] = metadata
                
            except Exception as meta_e:
                logger.warning(f"Metadata error for obs {obs_id} row {idx+1}: {meta_e}")
        
        # Process image entry
        image_entry = {'filename': img_filename, 'paths': {}, 'caption': None}
        img_found = False
        
        # Find images for each resolution siguiendo la estructura del repositorio
        for res in self.target_resolutions:
            res_path = self.image_path / split / res / img_filename
            if res_path.is_file():
                image_entry['paths'][res] = str(res_path)
                img_found = True
        
        if not img_found:
            continue
            
        # Process captions if available siguiendo la estructura del repositorio
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
                
        # Add the image entry to the observation
        if obs_id in data['observations']:
            data['observations'][obs_id]['image_entries'].append(image_entry)
            
    # Remove observations without valid images
    orig_obs_count = len(data['observations'])
    data['observations'] = {o: d for o, d in data['observations'].items() if d.get('image_entries')}
    if orig_obs_count != len(data['observations']):
        logger.info(f"Removed {orig_obs_count - len(data['observations'])} observations without images.")
        
    return data

def _create_structured_prompt(self, obs_data):
    """Crea un prompt estructurado combinando captions y metadatos para proceso multimodal"""
    # Inicializar las partes del prompt
    prompt_parts = []
    
    # Tarea principal - clara y directa
    prompt_parts.append("Identify this fungi species.")
    
    # Combinar todas las descripciones/captions disponibles
    all_captions = []
    for img_entry in obs_data.get('image_entries', []):
        if img_entry.get('caption'):
            all_captions.append(img_entry['caption'])
    
    if all_captions:
        # Seleccionar la más descriptiva (usualmente la más larga)
        main_caption = max(all_captions, key=len)
        prompt_parts.append(f"Description: {main_caption}")
    
    # Añadir metadatos estructurados si están disponibles
    if 'metadata' in obs_data:
        metadata = obs_data['metadata']
        metadata_parts = []
        
        # Información ecológica importante
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
        
        # Añadir metadatos estructurados al prompt
        if metadata_parts:
            prompt_parts.append("\\n".join(metadata_parts))
    
    # Combinar todas las partes
    final_prompt = "\\n".join(prompt_parts)
    return final_prompt

def extract_features_multimodal(self, data):
    """Extracción de características con ENSEMBLE corregido para diferentes dimensiones"""
    logger.info(f"Extrayendo características con ensemble corregido de {len(self.models)} modelos...")
    
    # Verificar si tenemos modelo multimodal fine-tuneado
    use_multimodal_finetuned = (self.models['bioclip'].get('finetuned', False) and 
                               'multimodal_adapter' in self.models['bioclip'] and
                               'extract_multimodal' in self.models['bioclip'])

    if use_multimodal_finetuned:
        logger.info("Usando modelo BioCLIP multimodal fine-tuneado + DINOv2 ensemble.")
    else:
        logger.info("Usando modelos base BioCLIP + DINOv2 ensemble.")
    
    # Limpiar cache CUDA antes de comenzar
    torch.cuda.empty_cache()
    
    for obs_id, obs_data in tqdm(data['observations'].items(), 
                                desc=f"Extrayendo Features (Ensemble Corregido)"):
        all_img_embeds_obs = []
        all_multimodal_embeds_obs = []
        
        # Crear prompt estructurado para esta observación
        structured_prompt = self._create_structured_prompt(obs_data)
        obs_data['structured_prompt'] = structured_prompt
        
        # Guardar embeddings por resolución
        obs_data['image_embeddings_by_resolution'] = {}
        obs_data['multimodal_embeddings_by_resolution'] = {}
        
        # Para cada imagen en la observación
        for img_entry in obs_data.get('image_entries', []):
            img_embeds_entry = []
            multimodal_embeds_entry = []
            valid_resolutions = 0
            
            for resolution, img_path in img_entry.get('paths', {}).items():
                if resolution not in self.target_resolutions:
                    continue
                    
                try:
                    # Cargar imagen
                    img = Image.open(img_path).convert('RGB')
                    
                    # === SOLUCIÓN: PROCESAR CADA MODELO POR SEPARADO ===
                    bioclip_embedding = None
                    dinov2_embedding = None
                    bioclip_multimodal_embedding = None
                    dinov2_multimodal_embedding = None
                    
                    # PROCESAR BIOCLIP
                    if 'bioclip' in self.models:
                        try:
                            model_data = self.models['bioclip']
                            model = model_data['model']
                            preprocess = model_data['preprocess']
                            tokenizer = model_data['tokenizer']
                            
                            img_input = preprocess(img).unsqueeze(0).to(self.device)
                            
                            with torch.no_grad(), torch.cuda.amp.autocast():
                                if use_multimodal_finetuned:
                                    # Usar función de extracción multimodal fine-tuneada
                                    multimodal_embedding = model_data['extract_multimodal'](img, structured_prompt)
                                    bioclip_multimodal_embedding = multimodal_embedding.squeeze()
                                    
                                    # También obtener embedding de imagen pura
                                    image_features = model.encode_image(img_input)
                                    image_features = F.normalize(image_features, p=2, dim=-1)
                                    bioclip_embedding = image_features.squeeze().cpu().numpy()
                                    
                                else:
                                    # Procesamiento estándar BioCLIP
                                    image_features = model.encode_image(img_input)
                                    image_features = F.normalize(image_features, p=2, dim=-1)
                                    bioclip_embedding = image_features.squeeze().cpu().numpy()
                                    
                                    if structured_prompt:
                                        # Fusión simple sin fine-tuning
                                        text_tokens = tokenizer([structured_prompt]).to(self.device)
                                        text_features = model.encode_text(text_tokens)
                                        text_features = F.normalize(text_features, p=2, dim=-1)
                                        
                                        # Combinación simple
                                        img_weight = 1.0 - self.text_image_weight
                                        text_weight = self.text_image_weight
                                        
                                        multimodal_features = img_weight * image_features + text_weight * text_features
                                        multimodal_features = F.normalize(multimodal_features, p=2, dim=-1)
                                        bioclip_multimodal_embedding = multimodal_features.squeeze().cpu().numpy()
                                    else:
                                        bioclip_multimodal_embedding = bioclip_embedding
                        
                        except Exception as e:
                            logger.warning(f"Error con BioCLIP en {img_path}: {e}")
                    
                    # PROCESAR DINOV2
                    if 'dinov2' in self.models:
                        try:
                            model_data = self.models['dinov2']
                            model = model_data['model']
                            preprocess = model_data['preprocess']
                            
                            img_input = preprocess(img).unsqueeze(0).to(self.device)
                            
                            with torch.no_grad(), torch.cuda.amp.autocast():
                                embedding = model(img_input)
                                embedding = F.normalize(embedding, p=2, dim=-1)
                                dinov2_embedding = embedding.squeeze().cpu().numpy()
                                # DINOv2 no tiene capacidad multimodal, usar embedding de imagen
                                dinov2_multimodal_embedding = dinov2_embedding
                        
                        except Exception as e:
                            logger.warning(f"Error con DINOv2 en {img_path}: {e}")
                    
                    if bioclip_embedding is not None:
                        # Aplicar peso de resolución
                        res_weight = self.resolution_weights.get(resolution, 1.0)
                        
                        # Usar BioCLIP como embedding principal (especializado en biología)
                        weighted_image_embedding = bioclip_embedding * res_weight
                        weighted_multimodal_embedding = bioclip_multimodal_embedding * res_weight
                        
                        # Si tenemos DINOv2, podemos hacer late fusion en el nivel de predicción
                        if dinov2_embedding is not None:
                            obs_data.setdefault('dinov2_embeddings', []).append(dinov2_embedding * res_weight)
                        
                        img_embeds_entry.append(weighted_image_embedding)
                        multimodal_embeds_entry.append(weighted_multimodal_embedding)
                        
                        # Guardar embeddings específicos por resolución
                        if resolution not in obs_data['image_embeddings_by_resolution']:
                            obs_data['image_embeddings_by_resolution'][resolution] = []
                        obs_data['image_embeddings_by_resolution'][resolution].append(weighted_image_embedding)
                        
                        if resolution not in obs_data['multimodal_embeddings_by_resolution']:
                            obs_data['multimodal_embeddings_by_resolution'][resolution] = []
                        obs_data['multimodal_embeddings_by_resolution'][resolution].append(weighted_multimodal_embedding)
                        
                        valid_resolutions += 1
                        
                    else:
                        logger.warning(f"No se pudo procesar {img_path} con ningún modelo")
                
                except Exception as e:
                    logger.warning(f"Error general procesando img {img_path}: {e}")
                    torch.cuda.empty_cache()
            
            # Procesar embeddings de imagen
            if img_embeds_entry:
                total_weight = sum(
                    self.resolution_weights.get(res, 1.0) 
                    for res in img_entry.get('paths', {}).keys() 
                    if res in self.target_resolutions and valid_resolutions > 0
                ) or 1.0
                
                if total_weight > 1e-6:
                    all_img_embeds_obs.append(np.sum(img_embeds_entry, axis=0) / total_weight)
            
            # Procesar embeddings multimodales
            if multimodal_embeds_entry:
                total_weight = sum(
                    self.resolution_weights.get(res, 1.0) 
                    for res in img_entry.get('paths', {}).keys() 
                    if res in self.target_resolutions and valid_resolutions > 0
                ) or 1.0
                
                if total_weight > 1e-6:
                    all_multimodal_embeds_obs.append(np.sum(multimodal_embeds_entry, axis=0) / total_weight)
        
        # Calcular embedding promedio para toda la observación
        if all_img_embeds_obs:
            obs_data['image_embedding_observation_avg'] = np.mean(all_img_embeds_obs, axis=0)
            
            # Calcular promedios por resolución para voting ensemble
            for resolution in obs_data['image_embeddings_by_resolution']:
                if len(obs_data['image_embeddings_by_resolution'][resolution]) > 0:
                    obs_data['image_embeddings_by_resolution'][resolution] = np.mean(
                        obs_data['image_embeddings_by_resolution'][resolution], axis=0
                    )
        
        # Calcular embedding multimodal promedio para toda la observación
        if all_multimodal_embeds_obs:
            obs_data['multimodal_embedding_observation_avg'] = np.mean(all_multimodal_embeds_obs, axis=0)
            
            # Calcular promedios por resolución para voting ensemble
            for resolution in obs_data['multimodal_embeddings_by_resolution']:
                if len(obs_data['multimodal_embeddings_by_resolution'][resolution]) > 0:
                    obs_data['multimodal_embeddings_by_resolution'][resolution] = np.mean(
                        obs_data['multimodal_embeddings_by_resolution'][resolution], axis=0
                    )
        
        # Calcular promedio de DINOv2 si existe
        if 'dinov2_embeddings' in obs_data and obs_data['dinov2_embeddings']:
            obs_data['dinov2_embedding_avg'] = np.mean(obs_data['dinov2_embeddings'], axis=0)
    
    # Limpiar CUDA cache al final
    torch.cuda.empty_cache()
    
    # Eliminar observaciones sin embeddings
    original_count = len(data['observations'])
    data['observations'] = {o: d for o, d in data['observations'].items() 
                          if 'image_embedding_observation_avg' in d or 'multimodal_embedding_observation_avg' in d}
    if original_count != len(data['observations']):
        logger.info(f"Eliminadas {original_count - len(data['observations'])} observaciones sin embedding.")
        
    logger.info(f"Extracción completada. BioCLIP como principal, DINOv2 como auxiliar")
    return data

def update_ensemble_weights(self):
    """Actualiza los pesos del ensemble para mejor balance BioCLIP/DINOv2"""
    # Pesos optimizados para el ensemble
    self.model_ensemble_weights = {
        'bioclip': 1.4,   # Ligeramente mayor peso (especializado en biología)
        'dinov2': 1.2     # Peso menor pero significativo (features visuales generales)
    }
    
    logger.info(f"Pesos del ensemble actualizados: {self.model_ensemble_weights}")

def _extract_caption_features(self, caption_text):
    """Extracción mejorada de características de captions con jerarquía taxonómica"""
    if not caption_text or not isinstance(caption_text, str):
        return {}
    
    # Normalizar texto a minúsculas
    caption_text = caption_text.lower()
    
    # Inicializar diccionario de features
    features = {
        'color': [],
        'shape': [],
        'texture': [],
        'habitat': [],
        'taxonomic': [],
        'taxonomic_hierarchy': {},
        'color_location': [],
        'texture_location': []
    }
    
    # Buscar términos de características en el texto
    for feature_type, terms in self.fungi_feature_terms.items():
        for term in terms:
            if term in caption_text:
                features[feature_type].append(term)
    
    # Buscar términos taxonómicos
    for term in self.taxonomic_terms:
        if term in caption_text:
            features['taxonomic'].append(term)
    
    # Buscar jerarquía taxonómica
    # Ejemplo: "genus: amanita" o "family russulaceae"
    for level, terms in self.taxonomic_hierarchy.items():
        # Buscar patrones como "genus amanita" o "genus: amanita"
        pattern = fr'{level}[\\s:]+([a-z]+)'
        matches = re.findall(pattern, caption_text)
        if matches:
            features['taxonomic_hierarchy'][level] = matches[0]
        else:
            # Buscar términos directos del nivel
            for term in terms:
                if term in caption_text:
                    features['taxonomic_hierarchy'][level] = term
                    break
    
    # Buscar características por pares
    for pair_type, pairs in self.fungi_feature_pairs.items():
        for pair in pairs:
            if pair in caption_text:
                features[pair_type].append(pair)
    
    return features

def _calculate_caption_similarity(self, query_features, train_features):
    """Cálculo mejorado de similitud entre características de captions"""
    if not query_features or not train_features:
        return 0.0
    
    total_similarity = 0.0
    # Pesos para tipos de características
    feature_weights = {
        'color': 0.25,
        'shape': 0.35,
        'texture': 0.2,
        'habitat': 0.3,
        'taxonomic': 0.7,  # Mayor peso para términos taxonómicos
        'color_location': 0.4,  # Alto peso para características específicas
        'texture_location': 0.35,
        'taxonomic_hierarchy': 0.8  # Máximo peso para jerarquía
    }
    
    # Calcular similitud para características simples
    for feature_type in ['color', 'shape', 'texture', 'habitat', 'taxonomic', 
                        'color_location', 'texture_location']:
        # Conjuntos de términos para este tipo
        query_terms = set(query_features.get(feature_type, []))
        train_terms = set(train_features.get(feature_type, []))
        
        if not query_terms or not train_terms:
            continue
        
        # Similitud de Jaccard
        intersection = len(query_terms.intersection(train_terms))
        union = len(query_terms.union(train_terms))
        
        if union > 0:
            jaccard_sim = intersection / union
            total_similarity += jaccard_sim * feature_weights.get(feature_type, 0.2)
    
    # Calcular similitud para jerarquía taxonómica (tratamiento especial)
    query_hierarchy = query_features.get('taxonomic_hierarchy', {})
    train_hierarchy = train_features.get('taxonomic_hierarchy', {})
    
    if query_hierarchy and train_hierarchy:
        # Verificar coincidencias en cada nivel
        hierarchy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        level_weights = {
            'kingdom': 0.1,  # Menos específico
            'phylum': 0.2,
            'class': 0.3,
            'order': 0.4,
            'family': 0.6,
            'genus': 1.0   # Más específico
        }
        
        hierarchy_sim = 0.0
        hierarchy_count = 0
        
        for level in hierarchy_levels:
            if level in query_hierarchy and level in train_hierarchy:
                if query_hierarchy[level] == train_hierarchy[level]:
                    hierarchy_sim += level_weights.get(level, 0.5)
                hierarchy_count += 1
        
        if hierarchy_count > 0:
            normalized_hierarchy_sim = hierarchy_sim / sum(level_weights.values())
            total_similarity += normalized_hierarchy_sim * feature_weights.get('taxonomic_hierarchy', 0.8)
    
    return total_similarity

# Añadir métodos a la clase
MultimodalFungiCLEF2025Pipeline.load_data = load_data
MultimodalFungiCLEF2025Pipeline._create_structured_prompt = _create_structured_prompt
MultimodalFungiCLEF2025Pipeline.extract_features_multimodal = extract_features_multimodal
MultimodalFungiCLEF2025Pipeline.update_ensemble_weights = update_ensemble_weights
MultimodalFungiCLEF2025Pipeline._extract_caption_features = _extract_caption_features
MultimodalFungiCLEF2025Pipeline._calculate_caption_similarity = _calculate_caption_similarity