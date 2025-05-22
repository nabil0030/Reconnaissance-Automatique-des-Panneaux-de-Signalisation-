import os
import pandas as pd
from PIL import Image
import numpy as np

# Chemins des dossiers
DATASET_FOLDER = "dataset/train"  # Dossier contenant les images d'entraînement
FILTERED_FOLDER = "dataset/train_filtered"  # Dossier pour les images filtrées
LABELS_FILE = "dataset/train_labels.csv"  # Fichier CSV des étiquettes

# Créer le dossier filtré
os.makedirs(FILTERED_FOLDER, exist_ok=True)

# Charger les étiquettes
labels_df = pd.read_csv(LABELS_FILE, sep=";")


# Fonction pour détecter le flou
def is_blurry(image_path, threshold=100):
    img = Image.open(image_path).convert("L")  # Convertir en niveaux de gris
    img_array = np.array(img)
    variance_of_laplacian = cv2.Laplacian(img_array, cv2.CV_64F).var()
    return variance_of_laplacian < threshold


# Filtrer les images
for _, row in labels_df.iterrows():
    img_path = os.path.join(DATASET_FOLDER, row["Filename"])
    class_id = row["ClassId"]

    # Vérifier si l'image existe
    if not os.path.exists(img_path):
        print(f"Image manquante: {img_path}")
        continue

    # Vérifier si l'image est floue
    if is_blurry(img_path):
        print(f"Image floue ignorée: {img_path}")
        continue

    # Copier l'image dans le dossier filtré
    output_class_folder = os.path.join(FILTERED_FOLDER, str(class_id))
    os.makedirs(output_class_folder, exist_ok=True)
    output_path = os.path.join(output_class_folder, row["Filename"])
    Image.open(img_path).save(output_path)
    print(f"Image copiée: {output_path}")

print("Filtrage terminé.")