"""
3. Indexación y Contexto
Pipeline Multimodal FungiCLEF 2025

Este archivo contiene:
- Construcción de índices FAISS para búsqueda eficiente
- Creación de prototipos y medoides de clase
- Modelo de contexto ecológico
- Clasificación jerárquica taxonómica
"""

def build_index(self, data):
    """Construye índice FAISS para búsqueda eficiente con soporte multimodal"""
    logger.info("Construyendo índices FAISS mejorados y prototipos...")
    
    # Construir índice para embeddings de imagen
    image_index_data = self._build_single_index(data, 'image_embedding_observation_avg', 'image_embeddings')
    
    # Construir índice para embeddings multimodales si están disponibles
    multimodal_index_data = None
    if self.use_multimodal_processing:
        multimodal_count = sum(1 for obs_data in data['observations'].values() 
                            if 'multimodal_embedding_observation_avg' in obs_data)
        
        if multimodal_count > 0:
            logger.info(f"Construyendo índice multimodal con {multimodal_count} observaciones...")
            multimodal_index_data = self._build_single_index(data, 'multimodal_embedding_observation_avg', 'multimodal_embeddings')
    
    # Construir prototipos (centroides) para cada clase
    logger.info("Construyendo prototipos de clase...")
    self._build_class_prototypes(data)
    
    return data, {"image": image_index_data, "multimodal": multimodal_index_data}

def _build_single_index(self, data, feature_key, index_type):
    """Construye un índice FAISS para un tipo específico de embedding"""
    embeddings, class_indices, observation_ids = [], [], []
    
    for obs_id, obs_data in data['observations'].items():
        if feature_key in obs_data:
            original_class_id = obs_data.get('original_class_id')
            
            # Verify class ID exists and is in our mapping
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
    logger.info(f"{index_type} - Embeddings: {embeddings_np.shape}, Class indices: {class_indices_np.shape}")
    
    # Check for NaN/Inf values
    if np.isnan(embeddings_np).any() or np.isinf(embeddings_np).any():
        logger.warning(f"WARNING: NaN or Inf values detected in {index_type}!")
        # Replace invalid values with zeros
        embeddings_np = np.nan_to_num(embeddings_np)
    
    # Create enhanced HNSW index
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
    
    # Configure search parameter
    index.hnsw.efSearch = self.hnsw_ef_search
    
    # Return index data
    return {
        'index': index,
        'embeddings': embeddings_np,
        'class_indices': class_indices_np,
        'observation_ids': observation_ids
    }

def _build_class_prototypes(self, data):
    """Construye prototipos (centroides) para cada clase"""
    # Inicializar diccionarios para prototipos
    data['image_class_prototypes'] = {}
    data['multimodal_class_prototypes'] = {}
    
    # Construir prototipos de imagen
    self._build_prototype_by_type(data, 'image_embedding_observation_avg', 'image_class_prototypes')
    
    # Construir prototipos multimodales si están disponibles
    if self.use_multimodal_processing:
        multimodal_count = sum(1 for obs_data in data['observations'].values() 
                            if 'multimodal_embedding_observation_avg' in obs_data)
        
        if multimodal_count > 0:
            self._build_prototype_by_type(data, 'multimodal_embedding_observation_avg', 'multimodal_class_prototypes')

def _build_prototype_by_type(self, data, feature_key, prototype_key):
    """Construye prototipos para un tipo específico de embedding"""
    # Agrupar embeddings por clase
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
    
    # Calcular centroides para cada clase
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
    
    logger.info(f"Prototipos de clase construidos para {len(data[prototype_key])} clases en {prototype_key}")

def _build_ecological_context_model(self, train_data):
    """Construye modelos de probabilidad para especies basados en hábitat, sustrato, región y mes"""
    logger.info("Construyendo modelo contextual ecológico...")
    
    # Inicializar contadores y diccionarios
    self.habitat_species = defaultdict(Counter)  # habitat -> {species_id: count}
    self.substrate_species = defaultdict(Counter)  # substrate -> {species_id: count}
    self.region_species = defaultdict(Counter)  # bioregion -> {species_id: count}
    self.month_species = defaultdict(Counter)  # month -> {species_id: count}
    
    # Contadores totales por contexto
    self.habitat_counts = Counter()  # habitat -> total_count
    self.substrate_counts = Counter()  # substrate -> total_count
    self.region_counts = Counter()  # bioregion -> total_count
    self.month_counts = Counter()  # month -> total_count
    
    # Estadísticas para logging
    habitat_found = 0
    substrate_found = 0
    region_found = 0
    month_found = 0
    
    # Recorrer los datos de entrenamiento
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
                habitat_found += 1
            
            # Procesar sustrato
            substrate = metadata.get('substrate', '').lower().strip()
            if substrate:
                self.substrate_species[substrate][species_id] += 1
                self.substrate_counts[substrate] += 1
                substrate_found += 1
            
            # Procesar región biogeográfica
            bioregion = metadata.get('biogeographicalRegion', '').lower().strip()
            if bioregion:
                self.region_species[bioregion][species_id] += 1
                self.region_counts[bioregion] += 1
                region_found += 1
            
            # Procesar mes
            month = metadata.get('month', -1)
            if month != -1 and 1 <= month <= 12:
                self.month_species[month][species_id] += 1
                self.month_counts[month] += 1
                month_found += 1
    
    # Calcular probabilidades (con suavizado Laplace)
    self.habitat_probs = defaultdict(dict)
    self.substrate_probs = defaultdict(dict)
    self.region_probs = defaultdict(dict)
    self.month_probs = defaultdict(dict)
    
    # Constante de suavizado
    alpha = 0.1
    num_classes = len(self.class_to_idx)
    
    # Calcular probabilidades para cada contexto
    for habitat, species_counts in self.habitat_species.items():
        total = self.habitat_counts[habitat]
        for species_id in self.class_to_idx.keys():
            count = species_counts[species_id]
            # Probabilidad suavizada
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
    
    logger.info(f"Modelo contextual ecológico construido:")
    logger.info(f"  - Hábitats: {len(self.habitat_species)} tipos, {habitat_found} observaciones")
    logger.info(f"  - Sustratos: {len(self.substrate_species)} tipos, {substrate_found} observaciones")
    logger.info(f"  - Regiones: {len(self.region_species)} tipos, {region_found} observaciones")
    logger.info(f"  - Distribución mensual: {len(self.month_species)} meses, {month_found} observaciones")

def _apply_ecological_context(self, base_predictions, obs_data, top_k=10):
    """Aplica el modelo contextual ecológico para reordenar las predicciones"""
    if not base_predictions or not hasattr(self, 'habitat_probs'):
        return base_predictions[:top_k]
    
    # Extraer información de contexto de la observación
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
    
    # Si no hay información contextual, devolver predicciones base
    if not context:
        return base_predictions[:top_k]
    
    # Calcular scores finales combinando predicciones base con contexto ecológico
    final_scores = {}
    
    for species_id, base_score in base_predictions:
        # Comenzar con el score base
        final_score = base_score
        
        # Aplicar factores de contexto
        context_boost = 1.0
        
        # Habitat
        if 'habitat' in context and context['habitat'] in self.habitat_probs:
            habitat_prob = self.habitat_probs[context['habitat']].get(species_id, 0.0)
            # Aplicar boost logarítmico para manejar valores pequeños mejor
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
        
        # Month (seasonal patterns)
        if 'month' in context and context['month'] in self.month_probs:
            month_prob = self.month_probs[context['month']].get(species_id, 0.0)
            if month_prob > 0:
                month_boost = 1.0 + self.context_weights['month'] * (1.0 + np.log10(max(month_prob * 10, 1.0)))
                context_boost *= month_boost
        
        # Aplicar boost de contexto al score base
        final_score *= context_boost
        
        final_scores[species_id] = final_score
    
    # Ordenar y devolver las top-k predicciones
    ranked_predictions = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_predictions[:top_k]

def _build_taxonomic_hierarchy(self, train_data):
    """Construye una estructura jerárquica taxonómica"""
    # Inicializar diccionarios para mapeo taxonómico
    self.species_to_taxonomy = {}  # category_id -> {level: name}
    self.taxonomy_to_species = {   # {level: {name: [category_ids]}}
        'genus': {},
        'family': {},
        'order': {},
        'class': {},
        'phylum': {}
    }
    
    # Recorrer datos de entrenamiento para construir la jerarquía
    for obs_id, obs_data in train_data['observations'].items():
        species_id = obs_data.get('original_class_id', -1)
        if species_id == -1:
            continue
            
        # Verificar que tenemos metadata
        if 'metadata' not in obs_data:
            continue
            
        metadata = obs_data['metadata']
        
        # Inicializar mapeo para esta especie si no existe
        if species_id not in self.species_to_taxonomy:
            self.species_to_taxonomy[species_id] = {}
        
        # Buscar directamente en los campos de metadata
        taxonomic_levels = ['genus', 'family', 'order', 'class', 'phylum']
        for tax_level in taxonomic_levels:
            if tax_level in metadata and metadata[tax_level]:
                tax_name = str(metadata[tax_level]).lower().strip()
                if not tax_name or tax_name == 'nan':
                    continue
                    
                # Guardar en mapeo especie->taxonomía
                self.species_to_taxonomy[species_id][tax_level] = tax_name
                
                # Guardar en mapeo taxonomía->especies
                if tax_name not in self.taxonomy_to_species[tax_level]:
                    self.taxonomy_to_species[tax_level][tax_name] = []
                
                if species_id not in self.taxonomy_to_species[tax_level][tax_name]:
                    self.taxonomy_to_species[tax_level][tax_name].append(species_id)
    
    logger.info(f"Construida jerarquía taxonómica con {len(self.species_to_taxonomy)} especies")

def _build_enhanced_prototypes(self, train_data):
    """
    Crea medoides para cada clase - el ejemplo más representativo de cada clase
    sin filtrado de outliers.
    """
    logger.info("Construyendo medoides de clase...")
    
    # Agrupar embeddings por clase
    class_embeddings = {}
    for obs_id, obs_data in train_data['observations'].items():
        species_id = obs_data.get('original_class_id', -1)
        if species_id == -1 or 'image_embedding_observation_avg' not in obs_data:
            continue
            
        if species_id not in class_embeddings:
            class_embeddings[species_id] = []
        
        class_embeddings[species_id].append({
            'obs_id': obs_id,
            'embedding': obs_data['image_embedding_observation_avg']
        })
    
    # Para almacenar los medoides
    train_data['class_medoids'] = {}
    
    total_classes = len(class_embeddings)
    classes_processed = 0
    
    for species_id, samples in class_embeddings.items():
        classes_processed += 1
        
        if len(samples) == 0:
            continue
            
        if len(samples) == 1:  # Para una sola muestra, es el medoide
            train_data['class_medoids'][species_id] = samples[0]['obs_id']
            continue
        
        # Convertir a array para cálculos más rápidos
        embeddings = np.array([s['embedding'] for s in samples])
        obs_ids = [s['obs_id'] for s in samples]
        
        # Calcular centroide
        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-6:
            centroid = centroid / norm
        
        # Encontrar el medoide (muestra más cercana al centroide)
        distances_to_centroid = [
            cosine(embeddings[i], centroid) 
            for i in range(len(embeddings))
        ]
        
        # Obtener el índice del medoide (muestra más cercana al centroide)
        medoid_idx = np.argmin(distances_to_centroid)
        train_data['class_medoids'][species_id] = obs_ids[medoid_idx]
    
    # Estadísticas para logging
    logger.info(f"Medoides identificados para {len(train_data['class_medoids'])} clases")
    
    return train_data

def _hierarchical_predict(self, base_predictions, obs_data, top_k=10):
    """Aplica clasificación jerárquica basada en taxonomía"""
    if not base_predictions:
        return []
    
    # Extraer taxonomía de la observación si está disponible
    query_taxonomy = {}
    
    if 'metadata' in obs_data:
        metadata = obs_data['metadata']
        
        # Extraer información taxonómica directamente
        taxonomic_levels = ['genus', 'family', 'order', 'class', 'phylum']
        for tax_level in taxonomic_levels:
            if tax_level in metadata and metadata[tax_level]:
                tax_name = str(metadata[tax_level]).lower().strip()
                if tax_name and tax_name != 'nan':
                    query_taxonomy[tax_level] = tax_name
    
    # Análisis taxonómico de las predicciones base
    taxa_votes = {
        'genus': Counter(),
        'family': Counter(),
        'order': Counter(),
        'class': Counter(),
        'phylum': Counter()
    }
    
    # Acumular "votos" para cada taxón basados en las predicciones base
    for species_id, score in base_predictions:
        if species_id in self.species_to_taxonomy:
            species_taxonomy = self.species_to_taxonomy[species_id]
            
            for level, taxon in species_taxonomy.items():
                # El peso del voto es proporcional al score de la predicción
                taxa_votes[level][taxon] += score
    
    # Normalizar los votos para cada nivel taxonómico
    normalized_taxa_votes = {}
    for level, votes in taxa_votes.items():
        if not votes:
            continue
            
        # Normalizar dividiendo por el máximo
        max_vote = max(votes.values()) if votes else 1
        if max_vote > 0:
            normalized_taxa_votes[level] = {taxon: vote/max_vote for taxon, vote in votes.items()}
    
    # Re-ranking de las predicciones usando información taxonómica
    final_scores = {}
    
    for species_id, base_score in base_predictions:
        # Comenzar con el score base
        final_score = base_score
        
        # Aplicar bonificaciones taxonómicas si tenemos información
        if species_id in self.species_to_taxonomy:
            species_taxonomy = self.species_to_taxonomy[species_id]
            
            for level, taxon in species_taxonomy.items():
                # Calcular bonificación basada en taxonomía de la consulta (si está disponible)
                if level in query_taxonomy and taxon == query_taxonomy[level]:
                    # Coincidencia exacta con la taxonomía de la consulta
                    tax_bonus = self.taxonomy_level_weights[level] * 2.0
                    final_score *= (1.0 + tax_bonus)
                
                # Calcular bonificación basada en los "votos" acumulados
                elif level in normalized_taxa_votes and taxon in normalized_taxa_votes[level]:
                    # La bonificación depende de cuán fuerte es el consenso para este taxón
                    consensus_strength = normalized_taxa_votes[level][taxon]
                    tax_bonus = self.taxonomy_level_weights[level] * consensus_strength
                    final_score *= (1.0 + tax_bonus)
        
        final_scores[species_id] = final_score
    
    # Ordenar y devolver las top-k predicciones
    ranked_predictions = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_predictions[:top_k]

# Añadir métodos a la clase
MultimodalFungiCLEF2025Pipeline.build_index = build_index
MultimodalFungiCLEF2025Pipeline._build_single_index = _build_single_index
MultimodalFungiCLEF2025Pipeline._build_class_prototypes = _build_class_prototypes
MultimodalFungiCLEF2025Pipeline._build_prototype_by_type = _build_prototype_by_type
MultimodalFungiCLEF2025Pipeline._build_ecological_context_model = _build_ecological_context_model
MultimodalFungiCLEF2025Pipeline._apply_ecological_context = _apply_ecological_context
MultimodalFungiCLEF2025Pipeline._build_taxonomic_hierarchy = _build_taxonomic_hierarchy
MultimodalFungiCLEF2025Pipeline._build_enhanced_prototypes = _build_enhanced_prototypes
MultimodalFungiCLEF2025Pipeline._hierarchical_predict = _hierarchical_predict