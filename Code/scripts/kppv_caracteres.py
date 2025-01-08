import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from composants_connexes_caracteres import extract_components, load_caractere_features

import time
start_time = time.time()

K = 3
SIMILARITY_THRESHOLD = 0.5

# Attribue une étiquette à chaque composant en utilisant k-ppv
def knn_classification(components, caractere_features, caractere_labels, k, similarity_threshold=SIMILARITY_THRESHOLD):
    components_tensor = tf.convert_to_tensor(components, dtype=tf.float32)
    caracteres_tensor = tf.convert_to_tensor(caractere_features, dtype=tf.float32)
    distances = tf.norm(components_tensor[:, None, :] - caracteres_tensor[None, :, :], axis=-1)
    labels = []
    for i in range(len(components)):
        nearest_neighbors_indices = tf.argsort(distances[i])[:k]
        nearest_distances = distances[i].numpy()[nearest_neighbors_indices.numpy()]
        min_distance = nearest_distances[0]
        if min_distance < similarity_threshold:
            neighbor_labels = [caractere_labels[idx] for idx in nearest_neighbors_indices.numpy()]
            label_counts = pd.Series(neighbor_labels).value_counts().idxmax()
            labels.append(label_counts)
        else:
            labels.append(None)  # Si la distance est trop grande, ignorer ce composant
    return labels

# Annote l'image avec les étiquettes et les contours des composants
def annotate_image(page_img, contours, labels, caractere_labels, caractere_colors):
    annotated_img = page_img.copy()
    for (x, y, w, h), label in zip(contours, labels):
        if label is not None:  # Si l'étiquette est valide
            idx = caractere_labels.index(label)
            color = caractere_colors[idx]
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(annotated_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated_img

# Traitement principal
page_path = "../data/caracteres/page.png"
page_img = cv2.imread(page_path)
if page_img is None:
    raise FileNotFoundError(f"Page image not found: {page_path}")

caractere_features, caractere_labels, caractere_colors = load_caractere_features()
components, contours = extract_components(page_img)

if components.size > 0:
    labels = knn_classification(components, caractere_features, caractere_labels, K)
    labeled_img = annotate_image(page_img, contours, labels, caractere_labels, caractere_colors)
    cv2.imwrite("../output/annotated_page.png", labeled_img)
    print("Image annotée sauvegardée sous 'annotated_page.png'")
else:
    print("Aucun composant détecté dans le page.")
    
    

print("--- %s seconds ---" % (time.time() - start_time))
