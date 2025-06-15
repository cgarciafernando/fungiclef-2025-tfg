"""
5. Pipeline Principal y Ejecución
Pipeline Multimodal FungiCLEF 2025

Este archivo contiene:
- Script principal de ejecución
- Función main() completa
- Ejemplo de uso del pipeline
- Configuración de rutas para el repositorio
"""

# Importar todos los componentes del pipeline
from pathlib import Path
import time

# Ejecutar los archivos de componentes (asegurarse de que estén importados)
# exec(open('1_configuracion_y_carga_modelos.py').read())
# exec(open('2_procesamiento_datos.py').read())
# exec(open('3_indexacion_y_contexto.py').read())
# exec(open('4_entrenamiento_y_prediccion.py').read())

def main():
    """
    Función principal que ejecuta todo el pipeline multimodal FungiCLEF 2025
    siguiendo la estructura del repositorio fungiclef-2025-tfg/
    """
    # --- Configuración de Rutas del Repositorio ---
    # Ajusta esta ruta según donde tengas clonado el repositorio
    BASE_PROJECT_PATH = Path("../")  # Asumiendo que ejecutas desde cuadernos/
    
    # Verificar estructura del repositorio
    EXPECTED_DIRS = [
        "dataset/metadata/FungiTastic-FewShot",
        "dataset/images/FungiTastic-FewShot", 
        "dataset/captions",
        "models",
        "predicciones",
        "resultados"
    ]
    
    print("🔍 Verificando estructura del repositorio...")
    for dir_path in EXPECTED_DIRS:
        full_path = BASE_PROJECT_PATH / dir_path
        if not full_path.exists():
            print(f"⚠️  Creando directorio faltante: {full_path}")
            full_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✅ {dir_path}")
    
    # Verificar que existen los archivos de datos principales
    required_files = [
        "dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv",
        "dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Val.csv",
        "dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Test.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = BASE_PROJECT_PATH / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("\n❌ ARCHIVOS FALTANTES:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📥 Por favor, descarga el dataset de FungiCLEF 2025 y colócalo en dataset/")
        print("📖 Consulta el README.md para instrucciones de descarga")
        return
    
    print(f"\n📁 Ruta del proyecto: {BASE_PROJECT_PATH.absolute()}")
    
    # --- Ejecución del Pipeline ---
    try:
        print("\n🚀 --- Iniciando Pipeline Multimodal FungiCLEF 2025 ---")
        
        start_time_total = time.time()
        
        # Inicializar pipeline con rutas del repositorio
        pipeline = MultimodalFungiCLEF2025Pipeline(
            BASE_PROJECT_PATH / "dataset",  # Apunta a la carpeta dataset/
            metadata_subdir='metadata/FungiTastic-FewShot',
            image_subdir='images/FungiTastic-FewShot', 
            caption_subdir='captions'
        )
        
        # 1. Preparación de datos con fine-tuning
        print("\n1️⃣ Preparando datos e índice multimodal...")
        start_prepare = time.time()
        train_data, val_data, index_data = pipeline.prepare_data_and_index()
        prepare_time = time.time() - start_prepare
        print(f"   ✅ Preparación completada en {prepare_time/60:.2f} minutos")
        
        if train_data and index_data:
            # 2. Evaluar en validación
            if val_data and val_data.get('observations'):
                print("\n2️⃣ Evaluando en conjunto de validación...")
                start_eval = time.time()
                try:
                    final_recall = pipeline.evaluate(train_data, val_data, index_data, k=5)
                    eval_time = time.time() - start_eval
                    print(f"   ✅ Evaluación completada en {eval_time/60:.2f} minutos")
                    print(f"   🎯 Recall@5 final en validación: {final_recall:.4f}")
                except Exception as eval_e:
                    print(f"   ❌ Error en evaluación: {eval_e}")
                    final_recall = 0.0
            
            # 3. Generar predicciones de test
            print("\n3️⃣ Generando predicciones para Test...")
            start_test = time.time()
            try:
                # Nombre del archivo de submission con timestamp
                timestamp = time.strftime("%Y%m%d-%H%M")
                model_id = "Multimodal_Pipeline"
                sub_file = f"submission_{model_id}_ValR{final_recall:.4f}_{timestamp}.csv"
                
                # Generar predicciones (se guardarán en predicciones/)
                results_df = pipeline.predict_test(train_data, index_data, output_file=sub_file, k=10)
                test_time = time.time() - start_test
                print(f"   ✅ Predicción de test completada en {test_time/60:.2f} minutos")
                print(f"   💾 Resultados guardados en: predicciones/{sub_file}")
                
                # Mostrar primeras filas de resultados
                if results_df is not None:
                    print("\n📊 Primeras 5 predicciones:")
                    print(results_df.head())
                    
                    # Guardar métricas en resultados/
                    save_results_summary(pipeline, final_recall, prepare_time, eval_time, test_time, timestamp)
                    
            except Exception as test_e:
                print(f"   ❌ Error en predicción de test: {test_e}")
                
        else:
            print("\n❌ No se pudo completar la preparación de datos e índice.")
        
        # Tiempo total
        total_time = time.time() - start_time_total
        print(f"\n🏁 --- Pipeline Multimodal completado en {total_time/60:.2f} minutos ---")
        
        # Mostrar resumen final
        print_final_summary(pipeline, final_recall, total_time)

    except Exception as e:
        print(f"\n💥 Error durante la ejecución del pipeline: {e}")
        import traceback
        traceback.print_exc()

def save_results_summary(pipeline, recall, prepare_time, eval_time, test_time, timestamp):
    """Guarda un resumen de resultados en la carpeta resultados/"""
    try:
        import json
        
        # Crear directorio de resultados
        results_dir = Path(pipeline.base_path).parent / "resultados"
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Resumen de la ejecución
        summary = {
            "timestamp": timestamp,
            "model_type": "Multimodal BioCLIP + DINOv2",
            "validation_recall_at_5": float(recall),
            "execution_times": {
                "data_preparation_minutes": round(prepare_time/60, 2),
                "evaluation_minutes": round(eval_time/60, 2),
                "test_prediction_minutes": round(test_time/60, 2),
                "total_minutes": round((prepare_time + eval_time + test_time)/60, 2)
            },
            "model_config": {
                "ensemble_weights": pipeline.ensemble_weights,
                "resolution_weights": pipeline.resolution_weights,
                "k_neighbors": pipeline.k_neighbors,
                "text_image_weight": pipeline.text_image_weight,
                "use_multimodal_processing": pipeline.use_multimodal_processing,
                "use_rare_species_boost": pipeline.use_rare_species_boost
            },
            "dataset_info": {
                "num_classes": pipeline.num_classes,
                "train_observations": len(pipeline.train_data_cache['observations']) if pipeline.train_data_cache else 0,
                "val_observations": len(pipeline.val_data_cache['observations']) if pipeline.val_data_cache else 0
            }
        }
        
        # Guardar resumen
        summary_file = results_dir / f"experiment_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"   📈 Resumen guardado en: {summary_file}")
        
    except Exception as e:
        print(f"   ⚠️  Error guardando resumen: {e}")

def print_final_summary(pipeline, recall, total_time):
    """Imprime un resumen final del pipeline"""
    print("\n" + "="*60)
    print("🎯 RESUMEN FINAL - Pipeline Multimodal FungiCLEF 2025")
    print("="*60)
    
    # Información del modelo
    models_loaded = list(pipeline.models.keys())
    print(f"🤖 Modelos cargados: {', '.join(models_loaded)}")
    
    if 'bioclip' in pipeline.models:
        is_finetuned = pipeline.models['bioclip'].get('finetuned', False)
        print(f"🔧 BioCLIP fine-tuning: {'✅ Aplicado' if is_finetuned else '❌ No aplicado'}")
    
    # Métricas principales
    print(f"\n📊 MÉTRICAS:")
    print(f"   • Recall@5 (Validación): {recall:.4f}")
    print(f"   • Tiempo total: {total_time/60:.2f} minutos")
    print(f"   • Clases identificadas: {pipeline.num_classes}")
    
    # Configuración utilizada
    print(f"\n⚙️  CONFIGURACIÓN:")
    print(f"   • Procesamiento multimodal: {'✅' if pipeline.use_multimodal_processing else '❌'}")
    print(f"   • Boost especies raras: {'✅' if pipeline.use_rare_species_boost else '❌'}")
    print(f"   • K vecinos: {pipeline.k_neighbors}")
    print(f"   • Peso texto/imagen: {pipeline.text_image_weight:.2f}")
    
    # Archivos generados
    print(f"\n📁 ARCHIVOS GENERADOS:")
    print(f"   • Modelo: models/bioclip_multimodal_finetuned.pt")
    print(f"   • Predicciones: predicciones/submission_*.csv")
    print(f"   • Logs: fungiclef_multimodal_pipeline.log")
    print(f"   • Resultados: resultados/experiment_summary_*.json")
    
    print("="*60)

def example_single_prediction():
    """
    Ejemplo de cómo usar el pipeline para hacer una predicción individual
    """
    print("\n🔬 EJEMPLO: Predicción individual")
    
    # Inicializar pipeline
    pipeline = MultimodalFungiCLEF2025Pipeline("../dataset")
    
    # Cargar datos ya procesados (asumiendo que ya se ejecutó el pipeline)
    if hasattr(pipeline, 'train_data_cache') and pipeline.train_data_cache:
        train_data = pipeline.train_data_cache
        index_data = pipeline.index_data_cache
        
        # Tomar una observación de ejemplo del conjunto de validación
        val_data = pipeline.load_data(split='val')
        val_data = pipeline.extract_features_multimodal(val_data)
        
        # Seleccionar primera observación
        obs_id = list(val_data['observations'].keys())[0]
        obs_data = val_data['observations'][obs_id]
        
        print(f"📝 Prediciendo para observación: {obs_id}")
        
        # Hacer predicción
        predictions = pipeline.predict_observation(obs_data, train_data, index_data, top_k=5)
        
        print("🎯 Top-5 predicciones:")
        for i, (class_id, score) in enumerate(predictions, 1):
            print(f"   {i}. Clase {class_id}: {score:.4f}")
        
        # Mostrar clase verdadera si está disponible
        true_class = obs_data.get('original_class_id')
        if true_class:
            print(f"✅ Clase verdadera: {true_class}")
            is_correct = true_class in [p[0] for p in predictions]
            print(f"🎯 Predicción correcta: {'✅ SÍ' if is_correct else '❌ NO'}")

def setup_repository_structure():
    """
    Función auxiliar para configurar la estructura del repositorio
    """
    print("🏗️  Configurando estructura del repositorio...")
    
    base_path = Path("../")  # Directorio padre (raíz del repo)
    
    # Estructura de directorios necesaria
    directories = [
        "dataset/metadata/FungiTastic-FewShot",
        "dataset/images/FungiTastic-FewShot/train/300p",
        "dataset/images/FungiTastic-FewShot/train/500p", 
        "dataset/images/FungiTastic-FewShot/train/720p",
        "dataset/images/FungiTastic-FewShot/train/fullsize",
        "dataset/images/FungiTastic-FewShot/val/300p",
        "dataset/images/FungiTastic-FewShot/val/500p",
        "dataset/images/FungiTastic-FewShot/val/720p", 
        "dataset/images/FungiTastic-FewShot/val/fullsize",
        "dataset/images/FungiTastic-FewShot/test/300p",
        "dataset/images/FungiTastic-FewShot/test/500p",
        "dataset/images/FungiTastic-FewShot/test/720p",
        "dataset/images/FungiTastic-FewShot/test/fullsize",
        "dataset/captions/train",
        "dataset/captions/val", 
        "dataset/captions/test",
        "cuadernos",
        "app",
        "models",
        "predicciones",
        "resultados",
        "docs"
    ]
    
    # Crear directorios
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Estructura del repositorio configurada")
    
    # Crear README básico si no existe
    readme_path = base_path / "README.md"
    if not readme_path.exists():
        readme_content = """# FungiCLEF 2025 - TFG

Pipeline multimodal para identificación de especies de hongos usando BioCLIP + DINOv2.

## Estructura del Proyecto

```
fungiclef-2025-tfg/
├── dataset/                    # 📊 Datos del dataset
├── cuadernos/                  # 📓 Notebooks de desarrollo  
├── app/                        # 🎯 Aplicación principal
├── models/                     # 🧠 Modelos entrenados
├── predicciones/               # 📈 Resultados de predicción
├── resultados/                 # 🏆 Métricas y análisis
└── docs/                       # 📖 Documentación
```

## Uso Rápido

```python
# Ejecutar pipeline completo
python cuadernos/5_pipeline_principal.py

# O ejecutar paso a paso
python cuadernos/1_configuracion_y_carga_modelos.py
python cuadernos/2_procesamiento_datos.py  
python cuadernos/3_indexacion_y_contexto.py
python cuadernos/4_entrenamiento_y_prediccion.py
```

## Requisitos

- Python 3.8+
- PyTorch 2.0+
- CUDA (recomendado)
- Dataset FungiCLEF 2025

Ver `requirements.txt` para dependencias completas.
"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print("📝 README.md creado")

if __name__ == "__main__":
    """
    Punto de entrada principal
    """
    import sys
    
    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1:
        if sys.argv[1] == "--setup":
            setup_repository_structure()
            return
        elif sys.argv[1] == "--example":
            example_single_prediction()
            return
        elif sys.argv[1] == "--help":
            print("""
🚀 Pipeline Multimodal FungiCLEF 2025

Uso:
  python 5_pipeline_principal.py           # Ejecutar pipeline completo
  python 5_pipeline_principal.py --setup   # Configurar estructura del repo
  python 5_pipeline_principal.py --example # Ejemplo de predicción individual
  python 5_pipeline_principal.py --help    # Mostrar esta ayuda

Estructura del repositorio:
  fungiclef-2025-tfg/
  ├── dataset/                 # Datos (descargar de FungiCLEF 2025)
  ├── cuadernos/              # Notebooks divididos (este archivo)
  ├── app/                    # Aplicación Gradio
  ├── models/                 # Modelos entrenados (.pt)
  ├── predicciones/           # Submissions (.csv)
  ├── resultados/            # Métricas (.json)
  └── docs/                  # Documentación

📖 Consulta el README.md para más información.
            """)
            return
    
    # Ejecución principal
    main()

# Ejemplo de configuración para diferentes entornos
class EnvironmentConfig:
    """Configuración para diferentes entornos de ejecución"""
    
    @staticmethod
    def local_development():
        """Configuración para desarrollo local"""
        return {
            "base_path": Path("../"),
            "device": "cuda" if torch.cuda.is_available() else "cpu", 
            "batch_size": 16,
            "epochs": 5,
            "debug": True
        }
    
    @staticmethod  
    def colab_environment():
        """Configuración para Google Colab"""
        return {
            "base_path": Path("/content/drive/MyDrive/fungiclef-2025/"),
            "device": "cuda",
            "batch_size": 32, 
            "epochs": 10,
            "debug": False
        }
    
    @staticmethod
    def server_environment():
        """Configuración para servidor con GPU potente"""
        return {
            "base_path": Path("/data/fungiclef-2025/"),
            "device": "cuda",
            "batch_size": 64,
            "epochs": 15, 
            "debug": False
        }

# Ejemplo de uso con configuración personalizada
def run_with_config(config_name="local"):
    """Ejecutar pipeline con configuración específica"""
    
    if config_name == "local":
        config = EnvironmentConfig.local_development()
    elif config_name == "colab":
        config = EnvironmentConfig.colab_environment()
    elif config_name == "server":
        config = EnvironmentConfig.server_environment()
    else:
        raise ValueError(f"Configuración '{config_name}' no reconocida")
    
    print(f"🔧 Usando configuración: {config_name}")
    print(f"📁 Ruta base: {config['base_path']}")
    print(f"💻 Dispositivo: {config['device']}")
    
    # Inicializar pipeline con configuración
    pipeline = MultimodalFungiCLEF2025Pipeline(config["base_path"] / "dataset")
    
    # Aplicar configuración
    pipeline.device = torch.device(config["device"])
    
    # Ejecutar con parámetros personalizados...
    # (resto de la ejecución usando los valores de config)
            else:
                print("\n2️⃣ No hay datos de validación disponibles para evaluación.")
                final_recall = 