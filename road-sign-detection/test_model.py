import tensorflow as tf
import numpy as np
from PIL import Image

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

# Fonction pour tester une image
def test_image(image_path):
    try:
        # Prétraiter l'image
        img = Image.open(image_path).resize((128, 128))  # Redimensionner à 128x128
        img_array = np.array(img) / 255.0  # Normaliser entre 0 et 1
        img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour le batch

        # Faire une prédiction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Classe prédite
        confidence = predictions[0][predicted_class]  # Niveau de confiance

        # Afficher les résultats
        print(f"Panneau détecté: {class_names[predicted_class]}")
        print(f"Confiance: {confidence:.2f}")
    except Exception as e:
        print(f"Erreur lors du traitement de l'image: {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacez "chemin/vers/votre/image.png" par le chemin de l'image à tester
    test_image("dataset/train_converted/00014/00000_00001.png")