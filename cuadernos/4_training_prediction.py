"""
4. Entrenamiento y Predicción
Pipeline Multimodal FungiCLEF 2025

Este archivo contiene:
- Fine-tuning multimodal del modelo BioCLIP
- Métodos de predicción multimodal
- Evaluación y generación de resultados
- Pipeline principal de ejecución
"""

def finetune_bioclip_multimodal(self, train_data, val_data,
                                epochs=10,              # ✅ WORKING NOTE: 10 epochs
                                batch_size=32,          
                                lr_model=3e-5,          # ✅ WORKING NOTE: 3e-5 
                                lr_adapter=2e-4,        # ✅ WORKING NOTE: 2e-4
                                beta=0.75):             # ✅ WORKING NOTE: 0.75
    """
    Fine-tuning multimodal según WORKING NOTE - Arquitectura y parámetros optimizados
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    import math
    from collections import Counter

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
                
                # Crear prompt estructurado
                prompt = self.pipeline._create_structured_prompt(obs_data)
                
                # Buscar mejor imagen disponible
                best_resolution_found = False
                for img_entry in obs_data.get('image_entries', []):
                    for res in ['fullsize', '720p', '500p']:  # Priorizar resoluciones altas
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

    # Transformaciones restauradas a valores originales
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
    train_ds = FungiMultimodalDataset(train_data, transform=train_transform, 
                                      class_to_idx=self.class_to_idx, pipeline=self)
    val_ds = FungiMultimodalDataset(val_data, transform=self.models['bioclip']['preprocess'], 
                                    class_to_idx=self.class_to_idx, pipeline=self)

    if len(train_ds) == 0:
        logger.error("Dataset vacío")
        return None, 0.0

    # Weighted sampler restaurado a valores originales
    class_counts = Counter(train_ds.labels)
    max_count = max(class_counts.values())
    class_weights = [max(0.03, (max_count / class_counts[label]) ** (1/3)) for label in train_ds.labels]  
    weights = torch.DoubleTensor(class_weights)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=False)

    logger.info(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Modelo BioCLIP
    model = self.models['bioclip']['model'].to(self.device)
    tokenizer = self.models['bioclip']['tokenizer']

    
    # 1. Congelar TODO primero
    for param in model.parameters():
        param.requires_grad = False

    # 2. Descongelar las 6 ÚLTIMAS CAPAS COMPLETAS del transformer visual
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer') and hasattr(model.visual.transformer, 'resblocks'):
        total_layers = len(model.visual.transformer.resblocks)
        layers_to_unfreeze = 6  # 6 últimas capas
        
        logger.info(f"Total de capas en ViT: {total_layers}")
        logger.info(f"Descongelando las últimas {layers_to_unfreeze} capas")
        
        # Descongelar las últimas N capas
        for i in range(total_layers - layers_to_unfreeze, total_layers):
            logger.info(f"Descongelando capa {i}")
            for param in model.visual.transformer.resblocks[i].parameters():
                param.requires_grad = True
    
    # 3. Descongelar capas de normalización y proyección
    for name, param in model.named_parameters():
        # Descongelar layer norms
        if 'ln_' in name or 'norm' in name:
            param.requires_grad = True
        # Descongelar projection layer final
        if 'visual.proj' in name:
            param.requires_grad = True
            logger.info(f"Descongelando projection layer: {name}")

    # Verificar cuántos parámetros se descongelaron
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parámetros descongelados: {trainable_params:,} de {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    # === ADAPTER SEGÚN WORKING NOTE - ARQUITECTURA OPTIMIZADA ===
    num_classes = len(self.class_to_idx)
    
    # Implementación exacta del working note:
    # h1 = Dropout(0.3,GELU(LayerNorm(Linear(fused,2048))))
    # h2 = Dropout(0.4,GELU(LayerNorm(Linear(h1,1024))))  
    # o = Linear(h2,num_classes)
    multimodal_adapter = nn.Sequential(
        nn.Linear(model.visual.output_dim, 2048),  # ✅ WORKING NOTE: 2048
        nn.LayerNorm(2048),                        # ✅ WORKING NOTE: LayerNorm
        nn.GELU(),                                 # ✅ WORKING NOTE: GELU
        nn.Dropout(0.3),                          # ✅ WORKING NOTE: 0.3
        nn.Linear(2048, 1024),                    # ✅ WORKING NOTE: 1024
        nn.LayerNorm(1024),                       # ✅ WORKING NOTE: LayerNorm
        nn.GELU(),                                # ✅ WORKING NOTE: GELU
        nn.Dropout(0.4),                          # ✅ WORKING NOTE: 0.4
        nn.Linear(1024, num_classes)              # ✅ WORKING NOTE: output layer
    ).to(self.device)

    logger.info(f"Adapter creado con {sum(p.numel() for p in multimodal_adapter.parameters()):,} parámetros")

    # === OPTIMIZADOR SEGÚN WORKING NOTE ===
    model_params = [p for p in model.parameters() if p.requires_grad]
    adapter_params = list(multimodal_adapter.parameters())
    
    if len(model_params) == 0:
        logger.warning("¡No hay parámetros del modelo para entrenar! Solo adapter.")
        optimizer = optim.AdamW(adapter_params, lr=lr_adapter, weight_decay=0.01)
    else:
        optimizer = optim.AdamW([
            {'params': model_params, 'lr': lr_model, 'weight_decay': 1e-6},      # ✅ WORKING NOTE: 3e-5
            {'params': adapter_params, 'lr': lr_adapter, 'weight_decay': 5e-5}   # ✅ WORKING NOTE: 2e-4
        ])
        logger.info(f"Optimizador WORKING NOTE - LR modelo: {lr_model}, LR adapter: {lr_adapter}")

    # === SCHEDULER SEGÚN WORKING NOTE - COSINE ANNEALING CON WARMUP ===
    warmup_steps = min(300, len(train_loader) * 2)  # ✅ WORKING NOTE: warmup más largo
    total_steps = epochs * len(train_loader)
    
    def lr_lambda_cosine_warmup(current_step):
        if current_step < warmup_steps:
            # Warmup lineal
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing - WORKING NOTE
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_cosine_warmup)

    # === FUNCIÓN DE PÉRDIDA SEGÚN WORKING NOTE - OptimizedFocalLoss ===
    class OptimizedFocalLoss(nn.Module):
        def __init__(self, alpha=0.3, gamma=1.5, label_smoothing=0.1):  # ✅ WORKING NOTE
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
    
    criterion = OptimizedFocalLoss(alpha=0.3, gamma=1.5, label_smoothing=0.1)  # ✅ WORKING NOTE

    logger.info("Iniciando fine-tuning SEGÚN WORKING NOTE...")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Beta: {beta}")
    logger.info(f"LR modelo: {lr_model}, LR adapter: {lr_adapter}")
    logger.info("Arquitectura: 2048→1024→clases con GELU + LayerNorm")
    logger.info("Loss: OptimizedFocalLoss(alpha=0.3, gamma=1.5, label_smoothing=0.1)")

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # === TRAINING ===
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
                
                # Fusión multimodal
                fused_features = beta * image_features + (1 - beta) * text_features
                
                # Clasificación
                outputs = multimodal_adapter(fused_features)
                loss = criterion(outputs, labels)
                
                # Backward
                loss.backward()
                # Gradient clipping más suave según working note
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)         # ✅ Más suave
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
        
        # === VALIDATION ===
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

    # Guardar el modelo en la carpeta models/ del repositorio
    from pathlib import Path
    models_dir = Path(self.base_path) / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    
    if best_model_state:
        model_path = models_dir / "bioclip_multimodal_finetuned.pt"
        torch.save(best_model_state, model_path)
        logger.info(f"✅ Modelo guardado en: {model_path}")
        logger.info(f"🎯 Mejor Val Acc: {best_val_acc:.4f}")
    else:
        logger.warning("❌ No se guardó ningún modelo")

    return model, best_val_acc

def _predict_multimodal(self, obs_data, train_data, index_data, top_k=10):
    """Método de predicción optimizado para Recall@5 que aprovecha embeddings multimodales"""
    
    # Verificar si tenemos datos multimodales
    has_multimodal = ('multimodal_embedding_observation_avg' in obs_data and 
                     index_data.get('multimodal') is not None)
    
    # Verificar si tenemos SAFE Recall@5 activo
    use_safe_recall5 = (hasattr(self.models['bioclip'], 'finetuned') and 
                        self.models['bioclip']['finetuned'] and 
                        'safe_classifier' in self.models['bioclip'])
    
    # Obtener predicciones basadas en imágenes
    image_predictions = self._predict_by_type(
        obs_data, 'image_embedding_observation_avg',
        train_data, index_data['image'], 
        'image_class_prototypes', top_k
    )
    
    # Si no tenemos multimodal, retornar predicciones de imagen
    if not has_multimodal:
        return image_predictions
        
    # Obtener predicciones multimodales
    multimodal_predictions = self._predict_by_type(
        obs_data, 'multimodal_embedding_observation_avg', 
        train_data, index_data['multimodal'],
        'multimodal_class_prototypes', top_k
    )
    
    # === COMBINACIÓN OPTIMIZADA PARA RECALL@5 ===
    combined_scores = {}
    
    if use_safe_recall5:
        # Con SAFE Recall@5: Priorizar embeddings multimodales proyectados
        logger.debug("Usando combinación optimizada para Recall@5 con SAFE")
        
        # Los embeddings multimodales SAFE están optimizados para ranking/recall
        mm_weight = 0.75  # Mayor peso para multimodal con SAFE (75%)
        img_weight = 0.25  # Menor peso para imagen (25%)
        
        # Combinar predicciones multimodales (prioritarias)
        for class_id, score in multimodal_predictions:
            if class_id not in combined_scores:
                combined_scores[class_id] = 0
            combined_scores[class_id] += score * mm_weight
            
        # Combinar predicciones de imagen (complementarias)
        for class_id, score in image_predictions:
            if class_id not in combined_scores:
                combined_scores[class_id] = 0
            combined_scores[class_id] += score * img_weight
            
    else:
        # Sin SAFE: Combinación balanceada tradicional
        logger.debug("Usando combinación estándar sin SAFE")
        
        img_weight = 0.55  # Dar ligeramente más peso a las imágenes (55%)
        mm_weight = 0.45   # Peso para multimodal (45%)
        
        # Combinar predicciones de imagen
        for class_id, score in image_predictions:
            if class_id not in combined_scores:
                combined_scores[class_id] = 0
            combined_scores[class_id] += score * img_weight
            
        # Combinar predicciones multimodales
        for class_id, score in multimodal_predictions:
            if class_id not in combined_scores:
                combined_scores[class_id] = 0
            combined_scores[class_id] += score * mm_weight
    
    # === BOOST ADICIONAL PARA RECALL@5 ===
    if use_safe_recall5:
        # Aplicar boost adicional basado en consenso entre ambos métodos
        consensus_boost = {}
        
        # Encontrar clases que aparecen en ambas predicciones (mayor confianza)
        image_classes = set(p[0] for p in image_predictions)
        multimodal_classes = set(p[0] for p in multimodal_predictions)
        consensus_classes = image_classes.intersection(multimodal_classes)
        
        # Boost para clases con consenso
        for class_id in consensus_classes:
            if class_id in combined_scores:
                combined_scores[class_id] *= 1.15  # 15% boost para consenso
        
        # Boost adicional para top multimodal predictions (están optimizadas para recall)
        top_multimodal_classes = set(p[0] for p in multimodal_predictions[:3])  # Top-3
        for class_id in top_multimodal_classes:
            if class_id in combined_scores:
                combined_scores[class_id] *= 1.1  # 10% boost adicional
    
    # Ordenar predicciones combinadas
    combined_predictions = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # === DIVERSIDAD PARA RECALL@5 ===
    # Asegurar diversidad en las predicciones para maximizar Recall@5
    if use_safe_recall5 and len(combined_predictions) > top_k:
        # Seleccionar top predictions con diversidad taxonómica
        final_predictions = []
        used_genera = set()  # Para evitar concentrarse en un solo género
        
        for class_id, score in combined_predictions:
            # Intentar obtener información taxonómica
            genus = None
            if hasattr(self, 'species_to_taxonomy') and class_id in self.species_to_taxonomy:
                genus = self.species_to_taxonomy[class_id].get('genus')
            
            # Si no tenemos muchas del mismo género, incluir
            if genus is None or used_genera.count(genus) < 2:
                final_predictions.append((class_id, score))
                if genus:
                    used_genera.add(genus)
                
                if len(final_predictions) >= top_k:
                    break
        
        # Si no llenamos top_k con diversidad, completar con las mejores restantes
        if len(final_predictions) < top_k:
            remaining = [(cid, s) for cid, s in combined_predictions 
                        if (cid, s) not in final_predictions]
            final_predictions.extend(remaining[:top_k - len(final_predictions)])
        
        return final_predictions[:top_k]
    