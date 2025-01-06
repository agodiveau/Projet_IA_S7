import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from composants_connexes import extract_components, load_luminaire_features  # Importer les fonctions du fichier composants_connexes

K = 3
SIMILARITY_THRESHOLD = 500  # Seuil de similarité (distance maximale) pour valider un luminaire

# Attribue une étiquette à chaque composant en utilisant k-ppv et applique un seuil de similarité
def knn_classification(components, luminaire_features, luminaire_labels, k, similarity_threshold=SIMILARITY_THRESHOLD):
    components_tensor = tf.convert_to_tensor(components, dtype=tf.float32)
    luminaires_tensor = tf.convert_to_tensor(luminaire_features, dtype=tf.float32)
    
    # Calcul des distances entre chaque composant et chaque luminaire
    distances = tf.norm(components_tensor[:, None, :] - luminaires_tensor[None, :, :], axis=-1)
    
    labels = []
    for i in range(len(components)):
        nearest_neighbors_indices = tf.argsort(distances[i])[:k]
        nearest_distances = distances[i].numpy()[nearest_neighbors_indices.numpy()]
        
        # Vérifier si la distance minimale est en dessous du seuil
        min_distance = nearest_distances[0]
        if min_distance < similarity_threshold:
            neighbor_labels = [luminaire_labels[idx] for idx in nearest_neighbors_indices.numpy()]
            label_counts = pd.Series(neighbor_labels).value_counts().idxmax()
            labels.append(label_counts)
        else: # Si la distance est trop grande, ignorer ce composant
            labels.append(None)
    
    return labels

# Annote l'image avec les étiquettes et les contours des composants
def annotate_image(plan_img, contours, labels, luminaire_labels, luminaire_colors):
    annotated_img = plan_img.copy()
    for (x, y, w, h), label in zip(contours, labels):
        if label is not None:  # Si l'étiquette est valide
            idx = luminaire_labels.index(label)
            color = luminaire_colors[idx]
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(annotated_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated_img

# Traitement principal
plan_path = "../data/plans/plan.png"
plan_img = cv2.imread(plan_path)
if plan_img is None:
    raise FileNotFoundError(f"Plan image not found: {plan_path}")

luminaire_features, luminaire_labels, luminaire_colors = load_luminaire_features()
components, contours = extract_components(plan_img)

if components.size > 0: # Si des composants sont détectés
    labels = knn_classification(components, luminaire_features, luminaire_labels, K)
    labeled_img = annotate_image(plan_img, contours, labels, luminaire_labels, luminaire_colors)
    cv2.imwrite("../output/annotated_plan.png", labeled_img)
    print("Image annotée sauvegardée sous 'annotated_plan.png'")
else:
    print("Aucun composant détecté dans le plan.")
