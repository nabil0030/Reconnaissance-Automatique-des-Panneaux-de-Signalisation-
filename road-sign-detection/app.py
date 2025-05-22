from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
MODEL_PATH = "model.h5"
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier téléchargé"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    # Sauvegarder le fichier téléchargé
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Prétraiter l'image
    img = Image.open(file_path).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prédire
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    # Renvoyer le résultat
    return jsonify({
        "class_name": class_names[predicted_class],
        "confidence": float(confidence)
    })

if __name__ == "__main__":
    app.run(debug=True)