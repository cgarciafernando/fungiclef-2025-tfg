"""
Utilidades y funciones auxiliares para FungiCLEF 2025
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pickle
import torch
import torch.nn as nn

def setup_logging():
    """Configura logging para la aplicación"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

class FungiSpeciesVisualizer:
    """Visualizador de especies y espacios vectoriales para FungiCLEF"""
    
    def __init__(self, pipeline_instance):
        self.pipeline = pipeline_instance
        self.logger = logging.getLogger(__name__)
    
    def get_species_info(self, class_id: int, obs_data: Dict = None) -> Dict:
        """Obtiene información de la especie"""
        info = {
            'name': f'Species_{class_id}',
            'genus': '',
            'family': '',
            'order': '',
            'class': '',
            'phylum': ''
        }
        
        try:
            # Intentar obtener información taxonómica del pipeline
            if hasattr(self.pipeline, 'species_to_taxonomy') and class_id in self.pipeline.species_to_taxonomy:
                taxonomy = self.pipeline.species_to_taxonomy[class_id]
                info.update({
                    'genus': taxonomy.get('genus', ''),
                    'family': taxonomy.get('family', ''),
                    'order': taxonomy.get('order', ''),
                    'class': taxonomy.get('class', ''),
                    'phylum': taxonomy.get('phylum', '')
                })
                
                # Si tenemos género, crear nombre más descriptivo
                if taxonomy.get('genus'):
                    info['name'] = f"{taxonomy['genus']}_sp_{class_id}"
                    
        except Exception as e:
            self.logger.debug(f"Error obteniendo info taxonómica para clase {class_id}: {e}")
        
        return info
    
    def create_predictions_gallery(self, predictions: List[Tuple]) -> List[Image.Image]:
        """Crea galería con imágenes de las especies predichas"""
        gallery_images = []
        
        for i, (class_id, score) in enumerate(predictions[:5]):
            try:
                # Intentar encontrar imagen representativa de la especie
                species_image = self._find_species_representative_image(class_id)
                
                if species_image is not None:
                    # Añadir overlay con información
                    annotated_image = self._add_prediction_overlay(species_image, class_id, score, i+1)
                    gallery_images.append(annotated_image)
                else:
                    # Crear placeholder
                    placeholder = self._create_species_placeholder(class_id, score, i+1)
                    gallery_images.append(placeholder)
                    
            except Exception as e:
                self.logger.warning(f"Error creando imagen de especie {i+1}: {e}")
                error_placeholder = self._create_error_placeholder(class_id, score, i+1)
                gallery_images.append(error_placeholder)
        
        return gallery_images
    
    def _find_species_representative_image(self, class_id: int) -> Optional[Image.Image]:
        """Busca una imagen representativa de la especie en los datos de entrenamiento"""
        try:
            if not hasattr(self.pipeline, 'train_data_cache') or not self.pipeline.train_data_cache:
                return None
                
            train_data = self.pipeline.train_data_cache
            
            # Buscar observaciones de esta clase
            for obs_id, obs_data in train_data.get('observations', {}).items():
                if obs_data.get('original_class_id') == class_id:
                    # Intentar cargar imagen de esta observación
                    for img_entry in obs_data.get('image_entries', []):
                        for resolution in ['500p', '720p', '300p', 'fullsize']:
                            if resolution in img_entry.get('paths', {}):
                                img_path = Path(img_entry['paths'][resolution])
                                if img_path.exists():
                                    try:
                                        img = Image.open(img_path).convert('RGB')
                                        # Redimensionar para galería
                                        img.thumbnail((250, 250), Image.Resampling.LANCZOS)
                                        return img
                                    except Exception:
                                        continue
                    break  # Solo tomar la primera observación encontrada
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error buscando imagen representativa para clase {class_id}: {e}")
            return None
    
    def _add_prediction_overlay(self, image: Image.Image, class_id: int, score: float, rank: int) -> Image.Image:
        """Añade overlay con información de predicción a la imagen"""
        try:
            # Crear figura matplotlib
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.imshow(image)
            ax.axis('off')
            
            # Información para overlay
            emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank-1]
            species_info = self.get_species_info(class_id)
            
            display_name = species_info['name']
            if display_name.startswith('Species_'):
                if species_info.get('genus'):
                    display_name = f"{species_info['genus']} sp."
                else:
                    display_name = f"Species {class_id}"
            
            # Texto del overlay
            overlay_text = f"{emoji}\n{display_name}\n{score:.3f}"
            
            # Añadir texto con fondo semitransparente
            ax.text(0.02, 0.98, overlay_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontweight='bold')
            
            plt.tight_layout()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            # Convertir a imagen PIL
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            result_image = Image.fromarray(buf)
            plt.close(fig)
            
            return result_image
            
        except Exception as e:
            self.logger.warning(f"Error añadiendo overlay: {e}")
            return image
    
    def _create_species_placeholder(self, class_id: int, score: float, rank: int) -> Image.Image:
        """Crea placeholder cuando no hay imagen disponible"""
        try:
            # Crear imagen base
            placeholder_img = Image.new('RGB', (250, 250), color='#f8f9fa')
            
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.imshow(placeholder_img)
            ax.axis('off')
            
            # Información de la especie
            emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank-1]
            species_info = self.get_species_info(class_id)
            
            display_name = species_info['name']
            if display_name.startswith('Species_'):
                if species_info.get('genus'):
                    display_name = f"{species_info['genus']} sp."
                else:
                    display_name = f"Species {class_id}"
            
            # Crear texto del placeholder
            placeholder_text = f"{emoji}\n🍄\n{display_name}\nScore: {score:.3f}"
            
            # Información taxonómica adicional
            tax_info = []
            if species_info.get('genus'):
                tax_info.append(f"Genus: {species_info['genus']}")
            if species_info.get('family'):
                tax_info.append(f"Family: {species_info['family']}")
            
            if tax_info:
                placeholder_text += f"\n\n{chr(10).join(tax_info)}"
            
            ax.text(0.5, 0.5, placeholder_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
                    fontweight='bold')
            
            plt.tight_layout()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            result_image = Image.fromarray(buf)
            plt.close(fig)
            
            return result_image
            
        except Exception as e:
            self.logger.error(f"Error creando placeholder: {e}")
            # Fallback: imagen simple
            return Image.new('RGB', (250, 250), color='#e9ecef')
    
    def _create_error_placeholder(self, class_id: int, score: float, rank: int) -> Image.Image:
        """Crea placeholder de error"""
        try:
            error_img = Image.new('RGB', (250, 250), color='#ffebee')
            
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.imshow(error_img)
            ax.axis('off')
            
            emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank-1]
            error_text = f"{emoji}\n❌\nSpecies {class_id}\nScore: {score:.3f}\n\nError\ncargando imagen"
            
            ax.text(0.5, 0.5, error_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="mistyrose", alpha=0.8),
                    fontweight='bold')
            
            plt.tight_layout()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            result_image = Image.fromarray(buf)
            plt.close(fig)
            
            return result_image
            
        except Exception as e:
            self.logger.error(f"Error creando placeholder de error: {e}")
            return Image.new('RGB', (250, 250), color='#ffcdd2')
    
    def create_vector_space_plot(self, predictions: List[Tuple], query_image: str) -> Image.Image:
        """Crea visualización del espacio vectorial con embeddings reales si es posible"""
        try:
            return self._create_real_vector_space_plot(predictions, query_image)
        except Exception as e:
            self.logger.warning(f"Error creando plot vectorial real: {e}")
            return self._create_demo_vector_space_plot(predictions, query_image)
    
    def _create_real_vector_space_plot(self, predictions: List[Tuple], query_image: str) -> Image.Image:
        """Crea plot con embeddings reales usando PCA"""
        if not hasattr(self.pipeline, 'index_data_cache') or not self.pipeline.index_data_cache:
            raise ValueError("No hay datos de índice disponibles")
        
        index_data = self.pipeline.index_data_cache
        
        # Obtener embeddings
        if 'multimodal' in index_data and index_data['multimodal'] is not None:
            embeddings = index_data['multimodal']['embeddings']
            class_indices = index_data['multimodal']['class_indices']
            feature_type = "Multimodal (BioCLIP + texto + DINOv2)"
        elif 'image' in index_data and index_data['image'] is not None:
            embeddings = index_data['image']['embeddings']
            class_indices = index_data['image']['class_indices']
            feature_type = "Early Fusion (BioCLIP + DINOv2)"
        else:
            raise ValueError("No hay embeddings disponibles")
        
        # Aplicar PCA para reducir a 2D
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Crear plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Puntos de todas las muestras (más pequeños y transparentes)
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                  s=20, c='lightgray', alpha=0.3, label='Todas las muestras')
        
        # Destacar predicciones top
        pred_class_ids = [pred[0] for pred in predictions[:5]]
        pred_colors = ['red', 'orange', 'gold', 'lightcoral', 'pink']
        pred_sizes = [200, 150, 120, 100, 80]
        
        for i, (class_id, score) in enumerate(predictions[:5]):
            if class_id in self.pipeline.class_to_idx:
                class_idx = self.pipeline.class_to_idx[class_id]
                
                # Encontrar puntos de esta clase
                class_mask = class_indices == class_idx
                if np.any(class_mask):
                    class_points = embeddings_2d[class_mask]
                    
                    ax.scatter(class_points[:, 0], class_points[:, 1], 
                             s=pred_sizes[i], c=pred_colors[i], alpha=0.8,
                             label=f'#{i+1}: Species {class_id} ({score:.3f})',
                             edgecolors='black', linewidth=1)
        
        # Configurar plot
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')
        ax.set_title(f'Espacio Vectorial 2D - {feature_type}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir a imagen PIL
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return Image.fromarray(buf)
    
    def _create_demo_vector_space_plot(self, predictions: List[Tuple], query_image: str) -> Image.Image:
        """Crea visualización demo del espacio vectorial"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Generar datos sintéticos consistentes
            np.random.seed(hash(query_image) % 2**32)
            n_points = 200
            
            # Cluster principal
            center_x, center_y = np.random.randn(2) * 2
            cluster_x = np.random.randn(n_points) * 1.5 + center_x
            cluster_y = np.random.randn(n_points) * 1.5 + center_y
            
            # Puntos de fondo
            bg_x = np.random.randn(n_points // 2) * 4
            bg_y = np.random.randn(n_points // 2) * 4
            
            # Plot puntos de fondo
            ax.scatter(bg_x, bg_y, s=20, c='lightgray', alpha=0.4, label='Otras especies')
            ax.scatter(cluster_x[:-10], cluster_y[:-10], s=30, c='lightblue', alpha=0.6, label='Especies relacionadas')
            
            # Destacar predicciones
            pred_colors = ['red', 'orange', 'gold', 'lightcoral', 'pink']
            pred_sizes = [200, 150, 120, 100, 80]
            
            for i, (class_id, score) in enumerate(predictions[:5]):
                # Posición basada en score (más cerca del centro = mayor score)
                angle = i * 2 * np.pi / 5
                distance = (1 - score) * 2 + 0.5
                x = center_x + distance * np.cos(angle)
                y = center_y + distance * np.sin(angle)
                
                ax.scatter(x, y, s=pred_sizes[i], c=pred_colors[i], alpha=0.9,
                          label=f'#{i+1}: Species {class_id} ({score:.3f})',
                          edgecolors='black', linewidth=2)
            
            # Punto de query en el centro
            ax.scatter(center_x, center_y, s=300, c='lime', marker='*', 
                      label='Imagen Test', edgecolors='black', linewidth=2)
            
            # Configurar plot
            ax.set_xlabel('Componente Principal 1')
            ax.set_ylabel('Componente Principal 2')
            ax.set_title('Espacio Vectorial 2D - Early Fusion (Demo)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Añadir círculos de similitud
            for radius in [1, 2, 3]:
                circle = plt.Circle((center_x, center_y), radius, fill=False, 
                                  linestyle='--', alpha=0.3, color='gray')
                ax.add_patch(circle)
            
            plt.tight_layout()
            
            # Convertir a imagen PIL
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            
            return Image.fromarray(buf)
            
        except Exception as e:
            self.logger.error(f"Error creando plot demo: {e}")
            return self._create_empty_plot(f"Error: {str(e)}")
    
    def _create_empty_plot(self, message: str) -> Image.Image:
        """Crea plot vacío con mensaje"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, message, transform=ax.transAxes, fontsize=14,
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Espacio Vectorial 2D Early Fusion', fontsize=14)
        plt.tight_layout()
        
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return Image.fromarray(buf)
    
    def create_predictions_markdown(self, predictions: List[Tuple]) -> str:
        """Crea markdown con información detallada de predicciones"""
        if not predictions:
            return "# ❌ No se pudieron generar predicciones"
        
        lines = ["# 🏆 Predicciones del Modelo\n"]
        
        for i, (class_id, score) in enumerate(predictions[:5], 1):
            species_info = self.get_species_info(class_id)
            emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1]
            
            # Nombre de la especie
            species_name = species_info['name']
            if species_name.startswith('Species_'):
                if species_info.get('genus'):
                    display_name = f"*{species_info['genus']}* sp."
                else:
                    display_name = f"Species {class_id}"
            else:
                display_name = f"*{species_name}*"
            
            lines.append(f"## {emoji} {display_name}")
            lines.append(f"**Confianza:** {score:.4f}")
            
            # Barra de confianza visual
            confidence_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"**Nivel:** `{confidence_bar}` {score*100:.1f}%")
            
            # Información taxonómica
            tax_info = []
            if species_info.get('genus') and not species_name.startswith('Species_'):
                tax_info.append(f"**Género:** *{species_info['genus']}*")
            if species_info.get('family'):
                tax_info.append(f"**Familia:** *{species_info['family']}*")
            if species_info.get('order'):
                tax_info.append(f"**Orden:** *{species_info['order']}*")
            
            if tax_info:
                lines.extend(tax_info)
            
            # Categoría de confianza
            if score >= 0.8:
                lines.append("🟢 **Alta confianza**")
            elif score >= 0.5:
                lines.append("🟡 **Confianza media**")
            else:
                lines.append("🔴 **Baja confianza**")
            
            lines.append("")
        
        # Información adicional
        lines.append("---")
        lines.append("### 📊 Información del Análisis")
        lines.append(f"- **Modelo:** Early Fusion BioCLIP + DINOv2")
        lines.append(f"- **Especies evaluadas:** {len(predictions)} de {len(self.pipeline.class_to_idx) if hasattr(self.pipeline, 'class_to_idx') else 'N/A'}")
        lines.append(f"- **Score máximo:** {predictions[0][1]:.4f}")
        lines.append(f"- **Score mínimo:** {predictions[-1][1]:.4f}")
        
        return "\n".join(lines)

class ModelLoader:
    """Cargador de modelos fine-tuneados mejorado"""
    
    @staticmethod
    def load_finetuned_model(pipeline_instance, model_path: str = "models/bioclip_early_fusion_finetuned.pt"):
        """Carga modelo fine-tuneado con fallbacks robustos"""
        logger = logging.getLogger(__name__)
        
        # Lista de posibles ubicaciones del modelo
        base_path = Path(pipeline_instance.base_path) if hasattr(pipeline_instance, 'base_path') else Path(".")
        
        possible_paths = [
            base_path / model_path,
            base_path / "models" / "bioclip_early_fusion_finetuned.pt",
            base_path / "models" / "bioclip_multimodal_finetuned.pt",
            Path(model_path),
            Path("models") / "bioclip_early_fusion_finetuned.pt",
            Path("models") / "bioclip_multimodal_finetuned.pt"
        ]
        
        model_file = None
        for path in possible_paths:
            if path.exists():
                model_file = path
                break
        
        if model_file is None:
            logger.warning("Modelo fine-tuneado no encontrado. Usando modelo base BioCLIP.")
            return False
        
        try:
            logger.info(f"Cargando modelo fine-tuneado desde {model_file}")
            
            # Cargar checkpoint
            device = getattr(pipeline_instance, 'device', torch.device('cpu'))
            checkpoint = torch.load(model_file, map_location=device)
            
            # Aplicar estado del modelo base
            if 'model_state_dict' in checkpoint:
                # Filtrar parámetros para evitar errores de dimensión
                model_state = checkpoint['model_state_dict']
                current_model = pipeline_instance.models['bioclip']['model']
                
                # Cargar solo parámetros compatibles
                filtered_state = {}
                for name, param in model_state.items():
                    if name in current_model.state_dict():
                        if current_model.state_dict()[name].shape == param.shape:
                            filtered_state[name] = param
                        else:
                            logger.warning(f"Skipping parameter {name} due to shape mismatch")
                
                current_model.load_state_dict(filtered_state, strict=False)
                logger.info(f"Cargados {len(filtered_state)} parámetros del modelo base")
            
            # Crear y cargar adapter si está disponible
            if 'adapter_state_dict' in checkpoint:
                try:
                    num_classes = checkpoint.get('num_classes', len(pipeline_instance.class_to_idx))
                    expected_dim = checkpoint.get('input_dim', 512)
                    
                    # Crear adapter
                    multimodal_adapter = nn.Sequential(
                        nn.Linear(expected_dim, 1024),
                        nn.LayerNorm(1024),
                        nn.GELU(),
                        nn.Dropout(0.3),
                        nn.Linear(1024, 512),
                        nn.LayerNorm(512),
                        nn.GELU(),
                        nn.Dropout(0.4),
                        nn.Linear(512, num_classes)
                    ).to(device)
                    
                    # Cargar estado del adapter
                    multimodal_adapter.load_state_dict(checkpoint['adapter_state_dict'])
                    pipeline_instance.models['bioclip']['multimodal_adapter'] = multimodal_adapter
                    
                    logger.info(f"Adapter cargado para {num_classes} clases")
                    
                except Exception as adapter_e:
                    logger.warning(f"Error cargando adapter: {adapter_e}")
            
            # Crear función de extracción multimodal
            if 'beta' in checkpoint:
                beta = checkpoint['beta']
                
                def extract_multimodal_embeddings(img, prompt="A photograph of fungi species"):
                    """Extrae embeddings multimodales fusionados"""
                    try:
                        model = pipeline_instance.models['bioclip']['model']
                        preprocess = pipeline_instance.models['bioclip']['preprocess']
                        tokenizer = pipeline_instance.models['bioclip']['tokenizer']
                        
                        # Procesar imagen
                        if isinstance(img, str):
                            img = Image.open(img).convert('RGB')
                        
                        img_tensor = preprocess(img).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            # Embeddings de imagen
                            image_features = model.encode_image(img_tensor)
                            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                            
                            # Embeddings de texto
                            text_tokens = tokenizer([prompt]).to(device)
                            text_features = model.encode_text(text_tokens)
                            text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
                            
                            # Fusión multimodal
                            fused_features = beta * image_features + (1 - beta) * text_features
                            fused_features = torch.nn.functional.normalize(fused_features, p=2, dim=-1)
                            
                            return fused_features.cpu().numpy()
                    
                    except Exception as e:
                        logger.error(f"Error en extracción multimodal: {e}")
                        return None
                
                pipeline_instance.models['bioclip']['extract_multimodal'] = extract_multimodal_embeddings
                pipeline_instance.models['bioclip']['beta'] = beta
            
            # Marcar como fine-tuned
            pipeline_instance.models['bioclip']['finetuned'] = True
            
            # Log información del checkpoint
            logger.info(f"Modelo fine-tuneado cargado exitosamente:")
            logger.info(f"  - Época: {checkpoint.get('epoch', 'N/A')}")
            logger.info(f"  - Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
            logger.info(f"  - Beta: {checkpoint.get('beta', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo fine-tuneado: {e}")
            logger.info("Continuando con modelo base BioCLIP")
            return False

def load_cached_data(cache_dir: str = "pipeline_cache"):
    """Carga datos desde caché con validación robusta"""
    logger = logging.getLogger(__name__)
    
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        logger.info(f"Directorio de caché {cache_dir} no existe")
        return None, None, None, None
    
    try:
        # Definir archivos de caché
        files = {
            'train_data': cache_path / "train_data_cache.pkl",
            'val_data': cache_path / "val_data_cache.pkl", 
            'index_data': cache_path / "index_data_cache.pkl",
            'class_mappings': cache_path / "class_mappings.pkl"
        }
        
        # Verificar que todos los archivos existen
        missing_files = [name for name, path in files.items() if not path.exists()]
        if missing_files:
            logger.info(f"Archivos de caché faltantes: {missing_files}")
            return None, None, None, None
        
        # Verificar que los archivos no estén corruptos
        for name, path in files.items():
            if path.stat().st_size == 0:
                logger.warning(f"Archivo de caché vacío: {name}")
                return None, None, None, None
        
        logger.info("Cargando datos desde caché...")
        
        # Cargar cada archivo
        loaded_data = {}
        for name, path in files.items():
            try:
                with open(path, 'rb') as f:
                    loaded_data[name] = pickle.load(f)
                logger.debug(f"Cargado {name} desde caché")
            except Exception as e:
                logger.error(f"Error cargando {name}: {e}")
                return None, None, None, None
        
        # Validar datos cargados
        train_data = loaded_data['train_data']
        val_data = loaded_data['val_data']
        index_data = loaded_data['index_data']
        class_mappings = loaded_data['class_mappings']
        
        # Verificaciones básicas
        if not isinstance(train_data, dict) or 'observations' not in train_data:
            logger.warning("Datos de entrenamiento inválidos en caché")
            return None, None, None, None
        
        if not isinstance(class_mappings, dict) or not all(key in class_mappings for key in ['class_to_idx', 'idx_to_class', 'num_classes']):
            logger.warning("Mapeo de clases inválido en caché")
            return None, None, None, None
        
        if not isinstance(index_data, dict):
            logger.warning("Datos de índice inválidos en caché")
            return None, None, None, None
        
        logger.info(f"Datos cargados desde caché exitosamente:")
        logger.info(f"  - Observaciones train: {len(train_data.get('observations', {}))}")
        logger.info(f"  - Observaciones val: {len(val_data.get('observations', {})) if val_data else 0}")
        logger.info(f"  - Clases: {class_mappings.get('num_classes', 0)}")
        logger.info(f"  - Índices disponibles: {list(index_data.keys())}")
        
        return train_data, val_data, index_data, class_mappings
        
    except Exception as e:
        logger.error(f"Error cargando caché: {e}")
        return None, None, None, None

def save_cached_data(train_data, val_data, index_data, class_mappings, cache_dir: str = "pipeline_cache"):
    """Guarda datos en caché"""
    logger = logging.getLogger(__name__)
    
    try:
        cache_path = Path(cache_dir)
        cache_path.mkdir(exist_ok=True, parents=True)
        
        # Preparar datos para guardar
        cache_data = {
            'train_data': train_data,
            'val_data': val_data,
            'index_data': index_data,
            'class_mappings': class_mappings
        }
        
        # Guardar cada archivo
        for name, data in cache_data.items():
            file_path = cache_path / f"{name}_cache.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Guardado {name} en caché")
        
        logger.info(f"Datos guardados en caché en {cache_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error guardando caché: {e}")
        return False
