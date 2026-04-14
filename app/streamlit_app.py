import os
import pandas as pd
from PIL import Image
import streamlit as st
import sys

# Agregar la ruta raíz para importar correctamente desde src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.models.autoencoder import OutfitRecommenderAutoencoder
from src.models.siameses import OutfitRecommenderSiameses
from app.utils import (
    preprocess_input_data,
    get_tensor_tuple,
    reverse_transform,
    merge_with_links,
    display_images,
    get_categories_from_encoder,
    load_demo_examples
)

# Configuración de Streamlit
st.set_page_config(page_title="OUTFITMATCH", layout="wide")

# Rutas globales seguras
PATH_RESOURCES = os.path.join(BASE_DIR, "resources/")
PATH_DEMOS = os.path.join(BASE_DIR, "prendas_prueba_streamlit/")
EXPECTED_KEYS = ["gender", "subCategory", "articleType", "season", "usage", "Color"]

@st.cache_resource
def load_autoencoder():
    return OutfitRecommenderAutoencoder(PATH_RESOURCES)

@st.cache_resource
def load_siameses():
    return OutfitRecommenderSiameses(PATH_RESOURCES)

# Inicializar modelo para rellenar diccionarios dinámicamente
autoencoder_cache = load_autoencoder()
cat_dict = get_categories_from_encoder(autoencoder_cache)

# -------------------------- Interfaz Principal --------------------------
st.markdown("<h1 style='text-align: center;'>OUTFITMATCH</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Sistema de recomanació d'outfits amb Intel·ligència Artificial</p>", unsafe_allow_html=True)
st.write("---")

st.sidebar.markdown("### Ajustos i Dades")
model_option = st.sidebar.selectbox("Tria el model d'IA", ["Autoencoder", "Siameses"])
st.sidebar.write("---")

# Selector de Modo de Input
input_mode = st.sidebar.radio("Mètode d'Entrada", ["Utilitzar Exemple de Demostració", "Pujar la meva pròpia peça"])

df_data_prenda = None
uploaded_image = None
image_to_process = None
metadata_to_process = {k: (cat_dict[k][0] if k in cat_dict else "") for k in EXPECTED_KEYS}

if input_mode == "Utilitzar Exemple de Demostració":
    examples_dict = load_demo_examples(PATH_DEMOS)
    if not examples_dict:
        st.sidebar.error("No s'han trobat exemples a la carpeta.")
    else:
        selected_demo = st.sidebar.selectbox("Selecciona una peça de mostra", list(examples_dict.keys()))
        demo_data = examples_dict[selected_demo]
        image_to_process = Image.open(demo_data["image_path"])
        st.sidebar.image(image_to_process, caption=f"Exemple seleccionat: {selected_demo}", use_container_width=True)
        
        # Override pre-filled metadata
        for k in EXPECTED_KEYS:
            if k in demo_data["metadata"]:
                metadata_to_process[k] = demo_data["metadata"][k]

elif input_mode == "Pujar la meva pròpia peça":
    st.sidebar.markdown("#### Pas 1: Imatge")
    uploaded_image = st.sidebar.file_uploader("Puja la imatge de la teva peça", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image_to_process = Image.open(uploaded_image)
        st.sidebar.image(image_to_process, caption="Imatge d'entrada", use_container_width=True)

# Generación Dinámica de Selectboxes leyendo del IA Encoder
st.sidebar.markdown("#### Pas 2: Metadades de la Peça")
with st.sidebar.expander("🛠️ Revisa o Modifica les Metadades", expanded=(input_mode == "Pujar la meva pròpia peça")):
    for k in EXPECTED_KEYS:
        if k in cat_dict:
            options = cat_dict[k]
            default_val = metadata_to_process[k]
            idx = 0
            
            # Buscar coincidencia exacta o case insensitiva
            if default_val in options:
                idx = options.index(default_val)
            else:
                for i, opt in enumerate(options):
                    if opt.lower() == str(default_val).lower():
                        idx = i
                        break
            
            metadata_to_process[k] = st.selectbox(f"{k.capitalize()}", options, index=idx)

if image_to_process is not None:
    # Añadidmos un id Dummy para que el DataFrame no pete en funciones base
    metadata_to_process["id"] = "USER_INPUT" 
    df_data_prenda = pd.DataFrame([metadata_to_process])

# Renderización y procesado central
if image_to_process is None or df_data_prenda is None:
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.info("👈 Si us plau, utilitza el panell lateral per proporcionar la teva imatge i metadades per generar recomanacions.")

generate = st.sidebar.button("Generar Outfit", type="primary", use_container_width=True)

if generate and image_to_process is not None and df_data_prenda is not None:
    with st.spinner(f"Processant recomanacions amb {model_option}..."):
        try:
            if model_option == "Autoencoder":
                autoencoder_model = load_autoencoder()
                
                input_data = preprocess_input_data(df_data_prenda, autoencoder_model)
                input_data_drop = input_data.drop(["image", "id"], errors="ignore", axis=1)

                tensor_tuple = get_tensor_tuple(input_data_drop)
                input_embedding = autoencoder_model.get_embedding(image_to_process, tensor_tuple)
                indices, _ = autoencoder_model.iterative_max_score_selection(tensor_tuple, input_embedding)

                data_indices = autoencoder_model.catalog_tab_all[indices]
                df_outfit = pd.DataFrame(data_indices, columns=["id", "gender", "subCategory", "articleType", "season", "usage", "Color"])

                input_data = input_data.drop(["image"], errors="ignore", axis=1)
                df_outfit = pd.concat([input_data, df_outfit])
                df_outfit = reverse_transform(df_outfit, autoencoder_model)
                df_merged = merge_with_links(df_outfit, PATH_RESOURCES)
                
                st.success("🎉 Outfit trobat!")
                st.markdown("### Resultats - Autoencoder")
                with st.expander("🔗 Dades tabulars completes"):
                    st.dataframe(df_merged)

                display_images(df_merged)

            elif model_option == "Siameses":
                siameses_model = load_siameses()
                
                outfit, _, _ = siameses_model.recommend_outfit(df_data_prenda.iloc[0].drop(["image"], errors="ignore", axis=0), image_to_process)
                df_outfit = pd.DataFrame(outfit)
                df_merged = merge_with_links(df_outfit, PATH_RESOURCES)
                
                st.success("🎉 Outfit trobat!")
                st.markdown("### Resultats - Siameses")
                st.caption("⚠️ Algunes imatges poden diferir del tipus esperat (Dataset flaw).")
                
                with st.expander("🔗 Dades tabulars completes"):
                    st.dataframe(df_merged)

                display_images(df_merged)
                
        except Exception as e:
            st.error(f"S'ha produït un error inesperat: {e}")
