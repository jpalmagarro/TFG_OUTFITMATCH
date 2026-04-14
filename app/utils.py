import os
import glob
import json
import time
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from io import BytesIO
import tensorflow as tf

def get_categories_from_encoder(autoencoder_model):
    """
    Extrae las clases categóricas que la IA ha aprendido, parseando el LabelEncoder.
    Devuelve un diccionario con { 'columna': ['clase1', 'clase2', ...] }
    """
    categories = {}
    for col in autoencoder_model.le_tab:
        categories[col] = list(autoencoder_model.le_tab[col].classes_)
    return categories

def load_demo_examples(demo_folder):
    """
    Escanea la carpeta de demostración y devuelve un diccionario con 
    la estructura: { "ID Prenda": { "image_path": "rutaimg", "metadata": {...} } }
    """
    examples = {}
    if not os.path.exists(demo_folder):
        return examples
        
    for json_file in glob.glob(os.path.join(demo_folder, "*.json")):
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        jpg_file = os.path.join(demo_folder, f"{base_name}.jpg")
        
        if os.path.exists(jpg_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                examples[base_name] = {
                    "image_path": jpg_file,
                    "metadata": metadata
                }
            except Exception:
                pass
                
    return examples

def preprocess_input_data(input_data, autoencoder_model):
    input_data = input_data.copy(deep=True).map(lambda x: x.lower() if isinstance(x, str) else x)
    for col in autoencoder_model.le_tab:
        input_data[col] = autoencoder_model.le_tab[col].transform(input_data[col])
    return input_data

def get_tensor_tuple(input_data):
    return tuple(tf.convert_to_tensor(value, dtype=tf.int32) for value in input_data.values.flatten())

def reverse_transform(df, autoencoder_model):
    for col in autoencoder_model.le_tab:
        df[col] = autoencoder_model.le_tab[col].inverse_transform(df[col])
    return df

def merge_with_links(df_outfit, path_resources):
    df_link = pd.read_csv(os.path.join(path_resources, "url_catalog.csv"))
    df_link['id'] = df_link['filename'].str.replace('.jpg', '', regex=False)
    df_outfit['id'] = df_outfit['id'].astype(str)
    df_link['id'] = df_link['id'].astype(str)
    return pd.merge(df_outfit, df_link, on='id', how='inner')

def fetch_image_with_retries(url, retries=3, delay=1):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

def display_images(df, url_column='link', type_columns='subCategory', max_images=4, img_width=200):
    cols_per_row = 2
    for i in range(0, min(len(df), max_images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j >= len(df):
                break
            try:
                img = fetch_image_with_retries(df.iloc[i + j][url_column])
                tipo_cand = df.iloc[i + j][type_columns]
                if i == 0 and j == 0:
                    cols[j].image(img, caption=f"Entrada: {tipo_cand}", width=img_width)
                else:
                    cols[j].image(img, caption=f"Candidata {i + j}: {tipo_cand}", width=img_width)
            except Exception as e:
                cols[j].warning(f"Error imatge {i + j + 1}: {e}")
