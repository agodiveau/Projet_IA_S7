import cv2
import numpy as np

# Configuration
LUMINAIRES = [
    {"path": "../data/plans/catalogue/luminaire_01.png", "label": "Luminaire 01", "color": (0, 0, 255)},
    {"path": "../data/plans/catalogue/luminaire_03.png", "label": "Luminaire 03", "color": (0, 165, 255)},
    {"path": "../data/plans/catalogue/luminaire_07.png", "label": "Luminaire 07", "color": (255, 0, 255)},
    {"path": "../data/plans/catalogue/luminaire_10.png", "label": "Luminaire 10", "color": (0, 255, 255)},
    {"path": "../data/plans/catalogue/luminaire_11.png", "label": "Luminaire 11", "color": (255, 192, 203)},
    {"path": "../data/plans/catalogue/luminaire_15.png", "label": "Luminaire 15", "color": (255, 255, 0)},
]

HU_MOMENTS_SIZE = 7

# Charge et traite les images des luminaires en extrayant leurs caractéristiques de forme
def load_luminaire_features():
    luminaire_features = []
    labels = []
    colors = []
    for luminaire in LUMINAIRES:
        thresholded_img = preprocess_image(luminaire["path"])
        feature = extract_shape_features(thresholded_img)
        luminaire_features.append(feature)
        labels.append(luminaire["label"])
        colors.append(luminaire["color"])
    return np.array(luminaire_features), labels, colors

# Charge une image, applique le seuillage et la convertit en contours
def preprocess_image(image_path, threshold=100):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    _, thresholded = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded

# Supprime les pixels gris dans une image couleur en les remplaçant par du blanc
def remove_gray_pixels(image, delta=15):
    diff = np.max(image, axis=2) - np.min(image, axis=2)
    gray_mask = diff <= delta
    cleaned_image = image.copy()
    cleaned_image[gray_mask] = [255, 255, 255]
    return cleaned_image

# Améliore l'image pour binarisation
def preprocess_image_for_thresholding(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(image)
    gray_blurred = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
    return gray_blurred

# Détecte les composants dans le plan et retourne leurs contours et caractéristiques
def extract_components(plan_img, min_area=300):
    cleaned_img = remove_gray_pixels(plan_img)
    gray_cleaned = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    gray_preprocessed = preprocess_image_for_thresholding(gray_cleaned)
    threshold_img = cv2.adaptiveThreshold(
        gray_preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    threshold_img = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel)
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

# Extrait les caractéristiques de forme des contours d'une image
def extract_shape_features(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moments = []
    for contour in contours:
        if cv2.contourArea(contour) > 300:
            hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
            moments.append(hu_moments)
    
    if not moments:
        return None
    
    while len(moments) < HU_MOMENTS_SIZE:
        moments.append(np.zeros(7))
    
    return np.concatenate(moments)[:HU_MOMENTS_SIZE]
