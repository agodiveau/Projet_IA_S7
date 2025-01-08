import numpy as np
import tensorflow as tf
import cv2
from kppv import knn_classification, extract_components, load_luminaire_features

def get_number_classes(luminaire_features) :
    return len(luminaire_features)

def get_number_components(components) :
    return len(components)

def initialize_unconditional_probabilities(number_classes) :
    probability_belonging_to_Cj = []
    for i in range(number_classes):
        probability_belonging_to_Cj.append(1/number_classes)
    return probability_belonging_to_Cj


# Calcul des distances entre chaque composant et chaque luminaire
def calculate_distances_components_luminaires(components, luminaire_features) :
    components_tensor = tf.convert_to_tensor(components, dtype=tf.float32)
    luminaires_tensor = tf.convert_to_tensor(luminaire_features, dtype=tf.float32)
    distances = tf.norm(components_tensor[:, None, :] - luminaires_tensor[None, :, :], axis=-1)
    return distances

def get_class_index(label) :
    if label == "Luminaire 01" :
        return 0
    elif label == "Luminaire 03" :
        return 1
    elif label == "Luminaire 07" :
        return 2
    elif label == "Luminaire 10" :
        return 3
    elif label == "Luminaire 11" :
        return 4
    elif label == "Luminaire 15" :
        return 5


#Charger l'image du plan.
plan_path = "../data/plans/plan.png"
plan_img = cv2.imread(plan_path)

components, _ = extract_components(plan_img) #Récupérer les caractéristiques des composantes connexes détectées dans l'image.
luminaire_features, luminaire_labels, _ = load_luminaire_features() #Récupérer les caractéristiques des luminaires du catalogue.


#Phase 1 : Construction du réseau bayésien.

number_classes = get_number_classes(luminaire_features)
number_components = get_number_components(components)

print("Il y a", number_classes, "classes.")
print(number_components, "composantes connexes ont été détectées dans l'image.")

#Initialisation des probabilités inconditionnelles ou P(C_{j}).
proba_Cj = initialize_unconditional_probabilities(number_classes)

print("P(Cj) initiales :", proba_Cj)

#Phase 1 : Apprentissage pour déterminer les probabilités conditionnelles P(X_{i} | C_{j})

K = 3
labels_kppv = knn_classification(components, luminaire_features, luminaire_labels, K)

proba_Xi_and_Cj = [[0]*len(luminaire_features[0]) for i in range(len(luminaire_features))] # matrice de b lignes et m colonnes

seuils = [0.4, 0.1, 0.5, 0.07, 0.07, 0.0001]

"""for i in range(len(components)) :
    class_index = get_class_index(labels_kppv[i])

    for j in range(len(components[i])) :
        if (components[i][j] < seuils[j]) and (class_index != None) :
            proba_Xi_and_Cj[class_index][j] += 1

print("P(Xi et Cj) =", proba_Xi_and_Cj)"""


