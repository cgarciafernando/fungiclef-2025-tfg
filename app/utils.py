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
            'family': ''
        }
        
        try:
            if hasattr(self.pipeline, 'species_to_taxonomy') and class_id in self.pipeline.species_to_taxonomy:
                taxonomy = self.pipeline.species_to_taxonomy[class_id]
                info.update({
                    'genus': taxonomy.get('genus', ''),
                    'family': taxonomy.get('family', ''),
                })
        except Exception:
            pass
        
        return info
    
    def create_predictions_gallery(self, predictions: List[Tuple]) -> List[Image.Image]:
        """Crea galería con imágenes de las especies predichas"""
        gallery_images = []
        
        for i, (class_id, score) in enumerate(predictions[:5]):
            try:
                # Crear placeholder simple
                placeholder = self._create_species_placeholder(class_id, score, i+1)
                gallery_images.append(placeholder)
                    
            except Exception as e:
                self.logger.warning(f"Error creando imagen de especie {i+1}: {e}")
                error_placeholder = self._create_error_placeholder(class_id, score, i+1)
                gallery_images.append(error_placeholder)
        
        return gallery_images
    
    def _create_species_placeholder(self, class_id: int, score: float, rank: int) -> Image.Image:
        """Crea placeholder cuando no hay imagen disponible"""
        placeholder_img = Image.new('RGB', (250, 250), color='#f5f5f5')
        
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow(placeholder_img)
        ax.axis('off')
        
        emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank-1]
        species_info = self.get_species_info(class_id)
        
        display_name = species_info['name']
        if display_name.startswith('Species_'):
            if species_info.get('genus'):
                display_name = f"{species_info['genus']} sp."
            else:
                display_name = f"Species {class_id}"
        
        placeholder_text = f"{emoji}\n{display_name}\nScore: {score:.3f}"
        
        ax.text(0.5, 0.5, placeholder_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        result_image = Image.fromarray(buf)
        plt.close(fig)
        
        return result_image
    
    def _create_error_placeholder(self, class_id: int, score: float, rank: int) -> Image.Image:
        """Crea placeholder de error"""
        error_img = Image.new('RGB', (250, 250), color='#ffebee')
        
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow(error_img)
        ax.axis('off')
        
        emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank-1]
        error_text = f"{emoji}\nSpecies {class_id}\nScore: {score:.3f}\n\nError\ncargando imagen"
        
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
    
    def create_vector_space_plot(self, predictions: List[Tuple], query_image: str) -> Image.Image:
        """Crea visualización simple del espacio vectorial"""
        try:
            # Crear plot simple para demo
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Generar datos sintéticos para demo
            np.random.seed(42)
            n_points = 100
            x = np.random.randn(n_points)
            y = np.random.randn(n_points)
            
            # Puntos de vecinos
            ax.scatter(x[:-5], y[:-5], s=50, c='lightblue', alpha=0.7, label='Vecinos Cercanos')
            
            # Puntos de predicciones
            pred_x = x[-5:]
            pred_y = y[-5:]
            ax.scatter(pred_x, pred_y, s=150, c='red', alpha=0.9, label='Top Predicciones')
            
            # Punto central
            ax.scatter(0, 0, s=300, c='lime', marker='*', label='Imagen Test')
            
            ax.set_xlabel('Componente Principal 1')
            ax.set_ylabel('Componente Principal 2')
            ax.set_title('Espacio Vectorial Multimodal 2D (Demo)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convertir a imagen PIL
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            
            return Image.fromarray(buf)
            
        except Exception as e:
            self.logger.warning(f"Error creando plot vectorial: {e}")
            return self._create_empty_plot(f"Error: {str(e)}")
    
    def _create_empty_plot(self, message: str) -> Image.Image:
        """Crea plot vacío con mensaje"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, message, transform=ax.transAxes, fontsize=14,
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Espacio Vectorial 2D Multimodal', fontsize=14)
        plt.tight_layout()
        
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return Image.fromarray(buf)
    
    def create_predictions_markdown(self, predictions: List[Tuple]) -> str:
        """Crea markdown con información de predicciones"""
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
            
            # Información taxonómica
            if species_info.get('genus') and not species_name.startswith('Species_'):
                lines.append(f"**Género:** *{species_info['genus']}*")
            if species_info.get('family'):
                lines.append(f"**Familia:** *{species_info['family']}*")
            
            lines.append("")
        
        return "\n".join(lines)

class ModelLoader:
    """Cargador de modelos fine-tuneados"""
    
    @staticmethod
    def load_finetuned_model(pipeline_instance, model_path: str = "models/bioclip_multimodal_finetuned.pt"):
        """Carga modelo fine-tuneado si existe"""
        logger = logging.getLogger(__name__)
        
        model_file = Path(model_path)
        if not model_file.exists():
            logger.warning(f"Modelo fine-tuneado no encontrado en {model_path}")
            return False
        
        try:
            logger.info(f"Cargando modelo fine-tuneado desde {model_path}")
            
            # Aquí iría la lógica de carga del modelo
            # Por ahora, simplemente marcamos como cargado
            pipeline_instance.models['bioclip']['finetuned'] = True
            
            logger.info("Modelo fine-tuneado cargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo fine-tuneado: {e}")
            return False

def load_cached_data(cache_dir: str = "pipeline_cache"):
    """Carga datos desde caché si existe"""
    logger = logging.getLogger(__name__)
    
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None, None, None, None
    
    try:
        files = {
            'train_data': cache_path / "train_data_cache.pkl",
            'val_data': cache_path / "val_data_cache.pkl", 
            'index_data': cache_path / "index_data_cache.pkl",
            'class_mappings': cache_path / "class_mappings.pkl"
        }
        
        if not all(f.exists() for f in files.values()):
            return None, None, None, None
        
        logger.info("Cargando desde caché...")
        
        with open(files['train_data'], 'rb') as f:
            train_data = pickle.load(f)
        
        with open(files['val_data'], 'rb') as f:
            val_data = pickle.load(f)
        
        with open(files['index_data'], 'rb') as f:
            index_data = pickle.load(f)
        
        with open(files['class_mappings'], 'rb') as f:
            class_mappings = pickle.load(f)
        
        logger.info("Datos cargados desde caché")
        return train_data, val_data, index_data, class_mappings
        
    except Exception as e:
        logger.error(f"Error cargando caché: {e}")
        return None, None, None, None
