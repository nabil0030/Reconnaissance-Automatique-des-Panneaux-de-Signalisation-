import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Liste complète des classes
class_names = {
    0: "Limitation de vitesse (20 km/h)",
    1: "Limitation de vitesse (30 km/h)",
    2: "Limitation de vitesse (50 km/h)",
    3: "Limitation de vitesse (60 km/h)",
    4: "Limitation de vitesse (70 km/h)",
    5: "Limitation de vitesse (80 km/h)",
    6: "Fin de limitation de vitesse (80 km/h)",
    7: "Limitation de vitesse (100 km/h)",
    8: "Limitation de vitesse (120 km/h)",
    9: "Interdiction de dépassement",
    10: "Interdiction de dépassement pour véhicules > 3.5 tonnes",
    11: "Priorité à la prochaine intersection",
    12: "Route prioritaire",
    13: "Cédez le passage",
    14: "Stop",
    15: "Interdiction de circulation",
    16: "Interdiction de poids lourds",
    17: "Interdiction d'entrée",
    18: "Autres dangers",
    19: "Virage dangereux à gauche",
    20: "Virage dangereux à droite",
    21: "Double virage",
    22: "Chaussée déformée",
    23: "Route glissante",
    24: "Rétrécissement de chaussée",
    25: "Travaux",
    26: "Feu de signalisation",
    27: "Passage piétons",
    28: "Enfants",
    29: "Circulation à vélo",
    30: "Neige/glace",
    31: "Passage des animaux sauvages",
    32: "Fin de toutes les limitations",
    33: "Direction obligatoire à droite",
    34: "Direction obligatoire à gauche",
    35: "Tout droit",
    36: "Tout droit ou à droite",
    37: "Tout droit ou à gauche",
    38: "Contournement par la droite",
    39: "Contournement par la gauche",
    40: "Rond-point",
    41: "Fin d'interdiction de dépassement",
    42: "Fin d'interdiction de dépassement pour véhicules > 3.5 tonnes"
}

# Charger le modèle
model = tf.keras.models.load_model("model.h5")

# Paramètres
IMG_SIZE = (128, 128)

# Charger les étiquettes
labels_df = pd.read_csv("dataset/test_labels.csv", sep=";")

# Préparer les données
X_test = []
y_test = []

for _, row in labels_df.iterrows():
    img_path = f"dataset/test_converted/{row['Filename'].replace('.ppm', '.png')}"
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    X_test.append(img_array)
    y_test.append(row["ClassId"])

X_test = np.array(X_test)
y_test = np.array(y_test)

# Faire des prédictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Calculer la précision
accuracy = accuracy_score(y_test, predicted_classes)
print(f"Précision globale sur l'ensemble de test: {accuracy:.4f}")

# Exemple de quelques prédictions
for i in range(5):  # Afficher les 5 premières prédictions
    print(f"Image {i+1}:")
    print(f"Classe réelle: {class_names[y_test[i]]}")
    print(f"Classe prédite: {class_names[predicted_classes[i]]}")
    print(f"Confiance: {predictions[i][predicted_classes[i]]:.2f}")
    print("-" * 40)