from flask import Flask, render_template, request
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array

app = Flask(__name__)

class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def predict_disease(img_path):
    try:
        global model
        if 'model' not in globals():
            model = tf.keras.models.load_model('plant_disease_model.keras')

        img = load_img(img_path, target_size=(224,224))
        img_array = img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        return predicted_class, confidence
    except Exception as e:
        print(f"Error in predict_disease: {e}")
        return "Error", 0.0


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/upload", methods=["GET","POST"])
def upload():

    if request.method == "POST":
        try:
            file = request.files["file"]
            if not file:
                return render_template("result.html", prediction="No file uploaded", confidence=0, image_path=None)

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            disease, confidence = predict_disease(filepath)

            return render_template(
                "result.html",
                prediction=disease,
                confidence=round(confidence*100,2),
                image_path=filepath
            )
        except Exception as e:
            print(f"Error in upload route: {e}")
            return render_template("result.html", prediction=f"Internal error: {e}", confidence=0, image_path=None)
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)