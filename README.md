# OUTFITMATCH

Trabajo de Fin de Grado (TFG) - **OUTFITMATCH**.
Implementación de un sistema de recomendación de conjuntos de ropa (Outfits) basado en un catálogo de prendas utilizando Inteligencia Artificial.

**Autora:** Jill Areny Palma Garro

## Descripción
Esta aplicación web desarrollada en Streamlit permite subir la imagen de una prenda y sus metadatos asociados para generar automáticamente recomendaciones de _outfits_ (conjuntos completos) compatibles visual y categóricamente.

El motor de recomendación utiliza dos enfoques de IA basados en el aprendizaje de representaciones (embeddings): 
- Modelo **Autoencoder** 
- Modelo de **Redes Siamesas**.

## Estructura del Proyecto
- `app/`: Contiene la aplicación web interactiva (`streamlit_app.py`) y sus funciones base.
- `src/models/`: Aloja la arquitectura de recomendación y las clases de inferencia.
- `src/metrics/`: Scripts para evaluación y medición de la compatibilidad visual y lógica de los conjuntos.
- `docs/`: Memoria académica en PDF, presentación de defensa y material gráfico de muestra.
- `resources/`: Directorio necesario para la inferencia, que aloja los modelos pre-entrenados, tensores numpy o variables pickle requeridas. *(Nota: Por los límites de tamaño en repositorios pueden requerir descargas externas).*

## Requisitos e Instalación

1. Clona este repositorio y navega hasta su carpeta:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd TFG_OUTFITMATCH
   ```

2. (Opcional pero Recomendado) Crea un entorno virtual e instálalo:
   ```bash
   python -m venv env
   # Activa tu entorno (Windows):
   env\Scripts\activate
   # Activa tu entorno (Mac/Linux):
   source env/bin/activate
   ```

3. Instala todas las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

Asegúrate de tener la carpeta `resources/` en la raíz del proyecto con todos los archivos requeridos para que no salte error (los `.keras`, `.npy`, y `.pkl`).

Inicia la aplicación ejecutando:
```bash
streamlit run app/streamlit_app.py
```
Abre tu navegador en la URL que aparece en la terminal (usualmente `http://localhost:8501`).
