#!/usr/bin/env python3
"""
Interfaz Gradio para FungiCLEF 2025 - Sistema de Clasificación de Hongos
Aplicación web interactiva con visualización del espacio vectorial
"""

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import time
import logging
from typing import List, Tuple, Dict, Optional

from pipeline import MultimodalFungiCLEF2025Pipeline
from utils import FungiSpeciesVisualizer, setup_logging, load_cached_data, ModelLoader

# Configuración
BASE_PROJECT_PATH = "."  # Ruta relativa para deployment
logger = setup_logging()

class FungiCLEFDemoInterface:
    """Interfaz de demostración optimizada para FungiCLEF 2025"""
    
    def __init__(self, pipeline_instance):
        self.pipeline = pipeline_instance
        self.visualizer = FungiSpeciesVisualizer(pipeline_instance)
        self.demo_predictions_cache = {}
        self.demo_images_info = {}
        
        # Cargar imágenes de ejemplo
        self._load_demo_samples()
        logger.info("Interfaz de demo inicializada")
    
    def _load_demo_samples(self, max_samples: int = 50):
        """Carga imágenes de ejemplo para la demo"""
        try:
            test_data = self.pipeline.load_data(split='test')
            if not test_data or not test_data.get('observations'):
                logger.warning("No se pudieron cargar datos de test")
                return
            
            # Seleccionar muestra representativa
            test_images = []
            for obs_id, obs_data in test_data['observations'].items():
                for img_entry in obs_data.get('image_entries', []):
                    if img_entry.get('paths'):
                        test_images.append({
                            'obs_id': obs_id,
                            'filename': img_entry['filename'],
                            'paths': img_entry['paths'],
                            'caption': img_entry.get('caption'),
                            'obs_data': obs_data
                        })
            
            # Ordenar por calidad y tomar muestra
            test_images.sort(key=lambda x: (
                bool(x['caption']),
                'fullsize' in x['paths'] or '720p' in x['paths']
            ), reverse=True)
            
            selected_images = test_images[:max_samples]
            
            # Crear opciones para dropdown
            self.demo_image_options = []
            for img_data in selected_images:
                display_name = f"🔬 {img_data['filename'].replace('.jpg', '')}"
                if img_data['caption']:
                    display_name += " 📝"
                
                self.demo_image_options.append((display_name, img_data['filename']))
                self.demo_images_info[img_data['filename']] = img_data
                
            logger.info(f"Cargadas {len(self.demo_image_options)} imágenes de ejemplo")
            
        except Exception as e:
            logger.error(f"Error cargando muestras de demo: {e}")
            self.demo_image_options = []
    
    def predict_demo_image(self, selected_image: str) -> Tuple:
        """Realiza predicción y genera visualizaciones"""
        if not selected_image:
            return None, None, None, None, "❌ Selecciona una imagen"
        
        try:
            logger.info(f"Analizando: {selected_image}")
            
            # Verificar caché
            if selected_image in self.demo_predictions_cache:
                predictions, prediction_time = self.demo_predictions_cache[selected_image]
                logger.info("Usando predicción desde caché")
            else:
                # Obtener datos de la imagen
                img_info = self.demo_images_info.get(selected_image)
                if not img_info:
                    return None, None, None, None, f"❌ No se encontró: {selected_image}"
                
                # Crear observación para predicción
                obs_data = self._create_demo_observation(img_info)
                obs_data = self._extract_demo_features(obs_data)
                
                # Realizar predicción
                start_time = time.time()
                predictions = self.pipeline.predict_observation(
                    obs_data, 
                    self.pipeline.train_data_cache, 
                    self.pipeline.index_data_cache, 
                    top_k=5
                )
                prediction_time = time.time() - start_time
                
                # Guardar en caché
                self.demo_predictions_cache[selected_image] = (predictions, prediction_time)
                logger.info(f"Análisis completado en {prediction_time:.3f}s")
            
            # Cargar imagen principal
            main_image = self._load_demo_image(selected_image)
            if main_image is None:
                return None, None, None, None, f"❌ No se pudo cargar: {selected_image}"
            
            # Crear galería de predicciones
            prediction_gallery = self.visualizer.create_predictions_gallery(predictions)
            
            # Crear visualización del espacio vectorial
            vector_plot = self.visualizer.create_vector_space_plot(predictions, selected_image)
            
            # Crear información de predicciones
            predictions_text = self.visualizer.create_predictions_markdown(predictions)
            
            # Crear información detallada
            detailed_info = self._create_detailed_info(selected_image, predictions, prediction_time)
            
            return main_image, prediction_gallery, vector_plot, predictions_text, detailed_info
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            logger.error(error_msg)
            return None, None, None, None, error_msg
    
    def _create_demo_observation(self, img_info: Dict) -> Dict:
        """Crea observación compatible con el pipeline"""
        return {
            'original_class_id': -1,
            'image_entries': [{
                'filename': img_info['filename'],
                'paths': img_info['paths'],
                'caption': img_info.get('caption')
            }],
            'metadata': {}
        }
    
    def _extract_demo_features(self, obs_data: Dict) -> Dict:
        """Extrae características usando el pipeline"""
        try:
            temp_data = {'observations': {'demo_obs': obs_data}}
            temp_data = self.pipeline.extract_features_multimodal(temp_data)
            return temp_data['observations']['demo_obs']
        except Exception as e:
            logger.warning(f"Error extrayendo características: {e}")
            return obs_data
    
    def _load_demo_image(self, filename: str) -> Optional[Image.Image]:
        """Carga imagen de demo"""
        img_info = self.demo_images_info.get(filename)
        if not img_info:
            return None
        
        for resolution in ['fullsize', '720p', '500p', '300p']:
            if resolution in img_info['paths']:
                img_path = Path(img_info['paths'][resolution])
                if img_path.exists():
                    try:
                        return Image.open(img_path).convert('RGB')
                    except Exception:
                        continue
        return None
    
    def _create_detailed_info(self, filename: str, predictions: List[Tuple], 
                            prediction_time: float) -> str:
        """Crea información detallada del análisis"""
        img_info = self.demo_images_info.get(filename, {})
        
        info_lines = [
            f"# 🔬 {filename}",
            f"**⏱️ Tiempo de análisis:** {prediction_time:.3f}s",
            ""
        ]
        
        # Descripción si está disponible
        if img_info.get('caption'):
            info_lines.extend([
                "**📝 Descripción:**",
                f"{img_info['caption']}",
                ""
            ])
        
        # Top 5 predicciones con información taxonómica
        info_lines.append("## 🏆 Top 5 Especies Predichas")
        for i, (class_id, score) in enumerate(predictions[:5], 1):
            species_info = self.visualizer.get_species_info(class_id)
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
            
            info_lines.append(f"{emoji} **{display_name}** - {score:.4f}")
            
            # Información taxonómica
            if species_info.get('genus') and not species_name.startswith('Species_'):
                info_lines.append(f"   📋 Género: *{species_info['genus']}*")
            if species_info.get('family'):
                info_lines.append(f"   📋 Familia: *{species_info['family']}*")
        
        return "\n".join(info_lines)
    
    def create_gradio_interface(self) -> gr.Interface:
        """Crea la interfaz de Gradio"""
        
        with gr.Blocks(
            title="FungiCLEF 2025 - Demo Pipeline Multimodal",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1800px !important;
            }
            .gallery-item {
                border-radius: 8px;
            }
            """
        ) as interface:
            
            # Título principal
            gr.Markdown("""
            # 🍄 FungiCLEF 2025 - Demo Pipeline Multimodal
            **Sistema avanzado de clasificación de hongos con BioCLIP fine-tuned + DINOv2**
            """)
            
            with gr.Row():
                # Columna izquierda - Controles
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 Selección")
                    
                    image_dropdown = gr.Dropdown(
                        choices=self.demo_image_options,
                        label="Imagen de test",
                        value=self.demo_image_options[0][1] if self.demo_image_options else None,
                        interactive=True
                    )
                    
                    predict_btn = gr.Button(
                        "🔮 Analizar",
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.Markdown(f"""
                    **📊 Dataset:**
                    - {len(self.demo_image_options)} imágenes test
                    - Multimodal (imagen + texto)
                    
                    **⚡ Pipeline:**
                    - BioCLIP fine-tuned
                    - DINOv2 ensemble
                    - FAISS similarity search
                    """)
                
                # Columna central - Imagen principal
                with gr.Column(scale=1.2):
                    gr.Markdown("### 📸 Imagen a Analizar")
                    main_image = gr.Image(
                        label="",
                        show_label=False,
                        height=400
                    )
                
                # Columna derecha - Predicciones
                with gr.Column(scale=2.2):
                    gr.Markdown("### 🏆 Predicciones")
                    predictions_text = gr.Markdown(
                        value="Selecciona una imagen y haz clic en 'Analizar'",
                        height=400
                    )
            
            # Segunda fila - Galería y información
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🖼️ Ejemplos Visuales de Especies")
                    prediction_gallery = gr.Gallery(
                        label="",
                        show_label=False,
                        columns=5,
                        rows=1,
                        height=250,
                        object_fit="contain"
                    )
                
                with gr.Column(scale=3):
                    gr.Markdown("### 📊 Información Detallada")
                    detailed_info = gr.Markdown(
                        value="Aquí aparecerá información detallada del análisis",
                        height=250
                    )
            
            # Tercera fila - Espacio vectorial
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🧠 Espacio Vectorial Multimodal")
                    gr.Markdown("*Visualización de embeddings BioCLIP multimodales en 2D usando PCA*")
                    vector_plot = gr.Image(
                        label="",
                        show_label=False,
                        height=600
                    )
            
            # Conectar eventos
            predict_btn.click(
                fn=self.predict_demo_image,
                inputs=[image_dropdown],
                outputs=[main_image, prediction_gallery, vector_plot, predictions_text, detailed_info],
                show_progress=True
            )
            
            image_dropdown.change(
                fn=self.predict_demo_image,
                inputs=[image_dropdown],
                outputs=[main_image, prediction_gallery, vector_plot, predictions_text, detailed_info],
                show_progress=True
            )
        
        return interface


def main():
    """Función principal para lanzar la aplicación"""
    
    logger.info("Iniciando FungiCLEF 2025 Demo...")
    
    try:
        # Inicializar pipeline
        pipeline = MultimodalFungiCLEF2025Pipeline(BASE_PROJECT_PATH)
        
        # Intentar cargar modelo fine-tuneado
        ModelLoader.load_finetuned_model(pipeline)
        
        # Intentar cargar desde caché
        cached_data = load_cached_data()
        if cached_data[0] is not None:  # Si hay caché válido
            train_data, val_data, index_data, class_mappings = cached_data
            
            # Restaurar en pipeline
            pipeline.train_data_cache = train_data
            pipeline.val_data_cache = val_data
            pipeline.index_data_cache = index_data
            pipeline.class_to_idx = class_mappings['class_to_idx']
            pipeline.idx_to_class = class_mappings['idx_to_class']
            pipeline.num_classes = class_mappings['num_classes']
            
            logger.info("Datos cargados desde caché")
        else:
            # Cargar datos desde cero
            train_data, val_data, index_data = pipeline.prepare_data_and_index()
        
        if not pipeline.train_data_cache or not pipeline.index_data_cache:
            raise RuntimeError("Error inicializando pipeline")
        
        # Crear interfaz de demo
        demo_interface = FungiCLEFDemoInterface(pipeline)
        gradio_interface = demo_interface.create_gradio_interface()
        
        # Lanzar aplicación
        logger.info("Lanzando interfaz web...")
        gradio_interface.launch(
            share=False,  # Cambiar a True para tunnel público
            server_port=7860,
            show_error=True,
            inbrowser=True
        )
        
    except Exception as e:
        logger.error(f"Error iniciando aplicación: {e}")
        raise


if __name__ == "__main__":
    main()
