from flask import Flask, render_template, request, jsonify, session
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests  # Pour télécharger des images depuis une URL
import pymysql.cursors
from werkzeug.security import check_password_hash, generate_password_hash

# Connexion à la base de données MySQL
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',  # Par défaut avec XAMPP
        password='',  # Mot de passe vide par défaut
        database='road_sign_db',
        cursorclass=pymysql.cursors.DictCursor
    )

app = Flask(__name__)
app.secret_key = 'ton_clef_secrete'  # Clé secrète pour la session

MODEL_PATH = "final_model.h5"
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
def home():
    return render_template("home.html")

@app.route("/index")
def index_page():
    # Ici tu peux vérifier si l'utilisateur est connecté ou pas
    if "user_id" not in session:
        return redirect("/login")  # redirige vers login si pas connecté

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

@app.route("/predict_from_url", methods=["POST"])
def predict_from_url():
    data = request.get_json()
    if not data or "image_url" not in data:
        return jsonify({"error": "URL de l'image manquante"}), 400

    image_url = data["image_url"]

    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Impossible de télécharger l'image"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, "auto_downloaded_image.jpg")
        with open(file_path, "wb") as f:
            f.write(response.content)

        img = Image.open(file_path).resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]

        return jsonify({
            "class_name": class_names[predicted_class],
            "confidence": float(confidence)
        })
    except Exception as e:
        return jsonify({"error": f"Erreur lors du traitement : {str(e)}"}), 500

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        try:
            data = request.get_json(force=True, silent=True)
        except Exception:
            data = None

        if not data:
            data = request.form.to_dict()

        username = data.get("username")
        email = data.get("email")
        password = data.get("password")

        if not all([username, email, password]):
            return jsonify({"error": "Tous les champs sont requis"}), 400

        try:
            with get_db_connection() as connection:
                with connection.cursor() as cursor:
                    sql = "SELECT * FROM users WHERE username = %s OR email = %s"
                    cursor.execute(sql, (username, email))
                    result = cursor.fetchone()

                    if result:
                        return jsonify({"error": "Nom d'utilisateur ou email déjà utilisé"}), 400

                    hashed_password = generate_password_hash(password)
                    sql = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
                    cursor.execute(sql, (username, email, hashed_password))
                    connection.commit()
                    return jsonify({"message": "Inscription réussie!"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        try:
            data = request.get_json(force=True, silent=True)
        except Exception:
            data = None

        if not data:
            data = request.form.to_dict()

        username = data.get("username")
        password = data.get("password")

        if not all([username, password]):
            return jsonify({"error": "Identifiants manquants"}), 400

        try:
            with get_db_connection() as connection:
                with connection.cursor() as cursor:
                    sql = "SELECT * FROM users WHERE username = %s"
                    cursor.execute(sql, (username,))
                    user = cursor.fetchone()

                    if not user or not check_password_hash(user["password"], password):
                        return jsonify({"error": "Nom d'utilisateur ou mot de passe incorrect"}), 401

                    session["user_id"] = user["id"]
                    session["username"] = user["username"]

                    return jsonify({"message": "Connexion réussie", "user": user})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("login.html")

if __name__ == "__main__":
    app.run(debug=True)
