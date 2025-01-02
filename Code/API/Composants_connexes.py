import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger l'image en haute qualité
image_path = "../data/plans/plan.png"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Prétraitement de l'image
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 3. Détection des composantes connexes
num_labels, labels_im = cv2.connectedComponents(binary_image)

# 4. Dessiner les boîtes englobantes sur l'image originale
output_image = image.copy()  # Utilisez une copie de l'image originale pour dessiner
for label in range(1, num_labels):  # Ignorer le label 0 (arrière-plan)
    mask = (labels_im == label).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask)  # Boîte englobante
    cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 5. Afficher l'image avec matplotlib (pas de compression)
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))  # Conversion pour Matplotlib
plt.title("Détection des composantes connexes")
plt.axis("off")
plt.show()
