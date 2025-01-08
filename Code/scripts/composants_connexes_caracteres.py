import cv2
import numpy as np

# Configuration des caractères
CARACTERES = [
    {"path": "../data/caracteres/catalogue/2.png", "label": "2", "color": (0, 0, 255)},
    {"path": "../data/caracteres/catalogue/d.png", "label": "d", "color": (0, 165, 255)},
    {"path": "../data/caracteres/catalogue/I.png", "label": "I", "color": (255, 0, 255)},
    {"path": "../data/caracteres/catalogue/n.png", "label": "n", "color": (0, 255, 255)},
    {"path": "../data/caracteres/catalogue/o.png", "label": "o", "color": (255, 192, 203)},
    {"path": "../data/caracteres/catalogue/u.png", "label": "u", "color": (255, 255, 0)},
]

HU_MOMENTS_SIZE = 7

# Charge et traite les images des caractères en extrayant leurs caractéristiques de forme
def load_caractere_features():
    caractere_features = []
    labels = []
    colors = []
    for caractere in CARACTERES:
        thresholded_img = preprocess_image(caractere["path"])
        feature = extract_shape_features(thresholded_img)
        if feature is not None:  # Vérifie si les caractéristiques sont valides
            caractere_features.append(feature)
            labels.append(caractere["label"])
            colors.append(caractere["color"])
    return np.array(caractere_features), labels, colors

# Charge une image, applique le seuillage
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    _, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresholded

# Prétraitement pour binarisation
def preprocess_image_for_thresholding(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresholded

# Détecte les composants dans la page et retourne leurs contours et caractéristiques
def extract_components(page_img, min_area=300):
    gray_cleaned = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
    threshold_img = preprocess_image_for_thresholding(gray_cleaned)
    contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components = []
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            component_img = threshold_img[y:y+h, x:x+w]
            features = extract_shape_features(component_img)
            if features is not None:
                components.append(features)
                filtered_contours.append((x, y, w, h))
    return np.array(components), filtered_contours

# Extrait les caractéristiques de forme à partir des moments de Hu
def extract_shape_features(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for contour in contours:
        if cv2.contourArea(contour) > 300:
            hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
            features.extend(hu_moments)  # Ajoute tous les moments de Hu pour le contour

    # Assure que le vecteur de caractéristiques a une taille fixe
    features = features[:HU_MOMENTS_SIZE]  # Tronque si nécessaire
    while len(features) < HU_MOMENTS_SIZE:
        features.append(0)  # Complète avec des zéros si nécessaire

    return np.array(features) if features else None
