import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2
from io import BytesIO

# Configuration de la page
st.set_page_config(page_title="Reconnaissance de caractères manuscrits", layout="wide", page_icon="✍️")

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    try:
        return joblib.load('emnist_mlp_model2.joblib')
    except FileNotFoundError:
        st.error("Modèle non trouvé. Veuillez d'abord entraîner le modèle avec le dataset EMNIST.")
        return None

# Fonction pour mapper les chiffres/lettres aux classes EMNIST
def map_emnist_class_to_label(y):
    # Mapping selon le codage ASCII
    mapping = {}
    # Chiffres: 0-9 (classes 0-9)
    for i in range(10):
        mapping[i] = str(i)
    # Lettres majuscules: A-Z (classes 10-35)
    for i in range(26):
        mapping[i + 10] = chr(65 + i)  # 65 est le code ASCII pour 'A'
    # Lettres minuscules: a-z (classes 36-61)
    for i in range(26):
        mapping[i + 36] = chr(97 + i)  # 97 est le code ASCII pour 'a'
    
    # Convertir les classes numériques en étiquettes
    if isinstance(y, (list, np.ndarray)):
        return np.array([mapping.get(int(cl), "?") for cl in y])
    else:
        return mapping.get(int(y), "?")

# Fonction de prétraitement pour une seule image
def preprocess_image(image, for_display=False):
    # Convertir en niveaux de gris
    image = ImageOps.grayscale(image)
    
    # Redimensionner à 28x28
    image = image.resize((28, 28))
    
    # Convertir en tableau numpy
    img_array = np.array(image)
    
    # Inverser les couleurs si nécessaire (texte noir sur fond blanc -> texte blanc sur fond noir)
    if np.mean(img_array) > 128:
        img_array = 255 - img_array
    
    # Si l'image est pour l'affichage, la retourner simplement
    if for_display:
        return img_array
    
    # Pour EMNIST, rotation de 90 degrés dans le sens inverse des aiguilles d'une montre et inversion horizontale
    img_array = np.rot90(img_array, k=3)
    img_array = np.fliplr(img_array)
    
    # Normaliser
    img_array = img_array / 255.0
    
    # Aplatir l'image
    img_array = img_array.reshape(1, 28 * 28).astype('float32')
    
    return img_array

# Fonction pour segmenter et reconnaître plusieurs caractères
def recognize_text(image):
    # Convertir l'image PIL en tableau numpy
    img_array = np.array(ImageOps.grayscale(image))
    
    # Redimensionner si l'image est trop grande
    max_dimension = 1000
    height, width = img_array.shape
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_height, new_width = int(height * scale), int(width * scale)
        img_array = cv2.resize(img_array, (new_width, new_height))
    
    # Binarisation
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilatation pour améliorer la détection des contours
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # Recherche de contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours trop petits
    min_area = 25
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Trier les contours de gauche à droite
    bounding_boxes = [cv2.boundingRect(contour) for contour in filtered_contours]
    sorted_boxes = sorted(bounding_boxes, key=lambda x: x[0])
    
    # Image colorée pour l'affichage des contours
    img_with_contours = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in sorted_boxes:
        cv2.rectangle(img_with_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Créer une figure avec les caractères détectés
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
    ax.set_title("Caractères détectés")
    ax.axis('off')
    
    # Charger le modèle
    model = load_model()
    if model is None:
        return None, None
    
    # Reconnaître chaque caractère
    recognized_chars = []
    char_images = []
    
    for i, (x, y, w, h) in enumerate(sorted_boxes):
        # Extraire le caractère
        char_img = binary[y:y+h, x:x+w]
        
        # Ajouter un padding pour centrer le caractère
        padding = 4
        char_img_padded = np.pad(char_img, padding, mode='constant', constant_values=0)
        
        # Redimensionner à 28x28
        char_img_resized = cv2.resize(char_img_padded, (28, 28))
        
        # Sauvegarder l'image du caractère pour l'affichage
        char_images.append(char_img_resized)
        
        # Prétraiter pour la prédiction (rotation et inversion pour EMNIST)
        char_img_rotated = np.rot90(char_img_resized, k=3)
        char_img_flipped = np.fliplr(char_img_rotated)
        
        # Normaliser
        char_img_norm = char_img_flipped / 255.0
        
        # Aplatir l'image
        char_vector = char_img_norm.flatten().reshape(1, -1)
        
        # Prédire
        pred_label = model.predict(char_vector)[0]
        pred_char = map_emnist_class_to_label(pred_label)
        
        recognized_chars.append(pred_char)
    
    return ''.join(recognized_chars), fig, char_images, recognized_chars

import streamlit as st
import os

def main():
    
    st.sidebar.title("Reconnaissance de caractères manuscrits")

    # Centrer l'image et la mettre en forme circulaire via CSS
    st.sidebar.markdown("""
        <style>
            .circle-img {
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 150px;  /* Taille du logo */
                height: 150px;  /* Assurer que la hauteur est égale à la largeur pour obtenir un cercle */
                border-radius: 50%;  /* Applique le style circulaire */
            }
        </style>
    """, unsafe_allow_html=True)

    # Afficher l'image avec la classe CSS "circle-img" pour la rendre circulaire
    st.sidebar.image('textVisionLogo.png', 
                     caption="EMNIST Dataset", use_column_width=False, width=150) 
    
    st.sidebar.write("**Reconnaissance de caractères manuscrits avec EMNIST**")
    st.sidebar.header("À propos du modèle")
    st.sidebar.write("""
        Ce modèle de reconnaissance de caractères manuscrits est un réseau de neurones entraîné sur le dataset EMNIST (Extended MNIST).
        Il peut reconnaître:
        - Les chiffres (0-9)
        - Les lettres majuscules (A-Z)
        - Les lettres minuscules (a-z) quand elles diffèrent visuellement des majuscules
        
        Détails du modèle:
        - Type: Réseau de neurones MLPClassifier (scikit-learn)
        - Architecture: 784-200-100-47
        - Activation: ReLU
        - Optimiseur: Adam
        - Précision: ~85-90% sur le jeu de test
    """)
    
    st.sidebar.header("Comment utiliser")
    st.sidebar.write("""
        1. Choisissez le mode de reconnaissance (caractère unique ou texte)
        2. Téléchargez une image contenant des caractères manuscrits
        3. Le modèle prétraitera l'image et effectuera la reconnaissance
        4. Les résultats de la prédiction seront affichés
    """)
    
    st.sidebar.header("Conseils")
    st.sidebar.write("""
        - Utilisez des images avec un bon contraste
        - Les caractères doivent être bien séparés pour la reconnaissance de texte
        - Les meilleurs résultats sont obtenus avec des images nettes et bien centrées
    """)
    
    # Section principale
    st.title("Reconnaissance de caractères manuscrits avec EMNIST")
    
    # Choix du mode
    mode = st.radio("Sélectionnez le mode de reconnaissance", 
                    ["Caractère unique", "Texte (plusieurs caractères)"])
    
    # Upload de l'image
    uploaded_file = st.file_uploader("Téléchargez une image...", type=["png", "jpg", "jpeg"])
    
    # Charger le modèle
    model = load_model()
    
    if uploaded_file is not None and model is not None:
        # Afficher l'image originale
        image = Image.open(uploaded_file)
        
        # Redimensionner l'image si elle est trop grande pour l'affichage
        max_width = 600
        if image.width > max_width:
            ratio = max_width / image.width
            image = image.resize((max_width, int(image.height * ratio)))
        
        st.image(image, caption='Image téléchargée', use_column_width=False)
        
        # Traiter selon le mode sélectionné
        if mode == "Caractère unique":
            # Prétraitement
            processed_image_display = preprocess_image(image, for_display=True)
            processed_image = preprocess_image(image)
            
            # Prédiction
            predicted_class = model.predict(processed_image)[0]
            predicted_char = map_emnist_class_to_label(predicted_class)
            
            # Probabilités
            probas = model.predict_proba(processed_image)[0]
            top_indices = np.argsort(probas)[-5:][::-1]  # Top 5 prédictions
            top_probs = probas[top_indices]
            top_chars = [map_emnist_class_to_label(idx) for idx in top_indices]
            
            # Affichage des résultats
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Image prétraitée")
                st.image(processed_image_display, caption='Image normalisée', use_column_width=True)
            
            with col2:
                st.subheader("Résultat de la prédiction")
                st.markdown(f"<h1 style='text-align: center; font-size: 72px;'>{predicted_char}</h1>", unsafe_allow_html=True)
                st.write(f"Classe prédite: {predicted_class}")
                
                # Graphique des probabilités
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(top_chars, top_probs)
                ax.set_xlabel('Caractère')
                ax.set_ylabel('Probabilité')
                ax.set_title('Top 5 des prédictions')
                st.pyplot(fig)
        
        else:  # Mode texte
            # Reconnaissance de texte
            recognized_text, contours_fig, char_images, pred_chars = recognize_text(image)
            
            if recognized_text:
                st.subheader("Caractères détectés dans l'image")
                st.pyplot(contours_fig)
                
                st.subheader("Texte reconnu")
                st.markdown(f"<h2 style='text-align: center; font-size: 48px;'>{recognized_text}</h2>", unsafe_allow_html=True)
                
                # Affichage des caractères individuels
                st.subheader("Détail des caractères reconnus")
                cols = st.columns(min(len(char_images), 10))
                
                for i, (img, char) in enumerate(zip(char_images, pred_chars)):
                    col_idx = i % len(cols)
                    with cols[col_idx]:
                        st.image(img, caption=f"{char}", width=50)
                        if (i+1) % len(cols) == 0 and i+1 < len(char_images):
                            cols = st.columns(min(len(char_images) - i - 1, 10))
    
    # Informations sur l'application
    st.markdown("---")
    st.markdown("""
    ### À propos de cette application
    
    Cette application utilise un réseau de neurones entraîné sur le dataset EMNIST pour reconnaître des caractères manuscrits.
    
    **Fonctionnalités:**
    - Reconnaissance d'un seul caractère manuscrit
    - Reconnaissance de texte manuscrit (plusieurs caractères)
    - Prétraitement automatique des images
    - Affichage des probabilités de prédiction
    
    **Technologies utilisées:**
    - Python
    - Streamlit
    - scikit-learn
    - OpenCV
    - Matplotlib
    - NumPy
    - Pandas
    """)

if __name__ == "__main__":
    main()
