# 👔 OUTFITMATCH

<div align="center">
  <a href="https://outfitmatch.streamlit.app/"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit"></a>
  <a href="https://github.com/jpalmagarro/TFG_OUTFITMATCH"><img src="https://img.shields.io/badge/GitHub-Repo-181717?logo=github&style=flat" alt="GitHub Repo"></a>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
</div>

<div align="center">
  <h3>📑 <a href="./docs/OUTFITMATCH_JILL_ARENY_PALMA_GARRO_TFG_INFORME_FINAL.pdf">Download Full Academic Thesis Report (PDF)</a> 📑</h3>
</div>

<br>

**Bachelor's Thesis (TFG) - OUTFITMATCH**
An Artificial Intelligence-based clothing outfit recommendation system. 

**Author:** Jill Areny Palma Garro

## 📋 Overview
This web application, built with Streamlit, allows users to choose or upload an image of a clothing item along with its basic metadata to automatically generate visually and categorically compatible full outfit recommendations.

The recommendation engine leverages two deep learning approaches based on representation learning (embeddings):
- **Autoencoder Model**
- **Siamese Networks Model**

### ✨ Key Features & Technical Achievements
* **Dual Deep Learning Architectures**: Implementation, training, and objective comparison of two distinct Neural Networks: an **Autoencoder** framework to compress visual features into latent spaces, and a **Siamese Network** geared directly towards measuring visual semantic compatibility.
* **Iterative Outfit Generation Engine**: A continuous recommendation loop that algorithmically constructs full clothing ensembles (Topwear, Bottomwear, etc.) by balancing embedding distances against hard category restrictions.
* **Multidimensional Rule-based Validation**: A rigorous custom metrics system built from scratch to evaluate outfit consistency mathematically. It cross-references generated garments through complex 3D compatibility matrices (`autoencoder_compatibility.pkl`) and visualizes latent similarities via `t-SNE` clustering.
* **Real-time Web App Demonstrator**: A meticulously designed, interactive Streamlit application serving as the primary showcase. It seamlessly integrates inference mechanisms and dynamic visual metadata parsing, proving the project's viability for a production environment.

## 📁 Project Structure
- `app/`: Contains the interactive web application (`streamlit_app.py`) and UI/UX utility services (`utils.py`).
- `src/models/`: Houses the core recommendation logic, neural architectures, and embedding extraction mechanisms.
- `src/metrics/`: Scripts for evaluating and measuring the logical consistency and visual compatibility of the AI-generated clothing combinations.
- `docs/`: Academic thesis documentation (PDF), defense presentation material, and multimedia demos.
- `prendas_prueba_streamlit/`: Folder containing example garments to test the application natively without external images.
- `resources/`: Directory housing the pre-trained weights, numpy tensors, and pickle dictionaries required for successful inference. *(Note: Due to repository size limits, you might need to supply these externally if cloning fresh).*

## 🚀 Installation & Usage

1. Clone this repository and move into it:
   ```bash
   git clone https://github.com/jpalmagarro/TFG_OUTFITMATCH.git
   cd TFG_OUTFITMATCH
   ```

2. (Optional but Highly Recommended) Create and activate a Virtual Environment:
   ```bash
   python -m venv env
   
   # Activate on Windows:
   env\Scripts\activate
   
   # Activate on Mac/Linux:
   source env/bin/activate
   ```

3. Install the required dependencies securely:
   ```bash
   pip install -r requirements.txt
   ```

4. **Resource Setup:**
   Confirm that your `resources/` folder is placed exactly at the root of the project with all required `.keras`, `.npy`, and `.pkl` binaries.

5. Bootstrap the web application:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   *Your default web browser should automatically open at `http://localhost:8501`.*
