# 🌿 Plant Disease Detection using Deep Learning

AI-powered web application that detects plant leaf diseases from images using a trained deep learning model.
The model analyzes a plant leaf image and predicts the type of disease affecting the plant.

🚀 Live Demo:
👉 https://huggingface.co/spaces/ramji-4412/plant-disease-detection

---

# 📌 Project Overview

Plant diseases significantly reduce crop productivity. Early detection helps farmers take quick action and prevent crop loss.

This project uses Deep Learning and Computer Vision to automatically identify plant diseases from leaf images.

The system:

1. Takes an image of a plant leaf

2. Processes it using a trained neural network

3. Predicts the disease class

4. Displays the result to the user

---

# 🧠 Technologies Used

-> Python

-> TensorFlow / Keras

-> NumPy

-> OpenCV

-> Gradio (for web interface)

-> Hugging Face Spaces (for deployment)

---

# ⚙️ How It Works

User uploads a plant leaf image

Image is preprocessed

resized

normalized

The trained CNN model analyzes the image

The model predicts the disease category

The result is displayed on the web interface

---

# 📊 Model Details

Model Type: Convolutional Neural Network (CNN)

Framework: TensorFlow / Keras

Input Size: (224 × 224) image

Output: Disease classification

The model was trained on a dataset containing multiple plant leaf diseases.

# 📁 Project Structure

plant-disease-detection/

│

├── app.py                 # Main application file

├── static/

│   └── script.js

│   └── style.css

├──templates/

│   └── about.html

│   └── home.html

│   └── result.html

│   └── upload.html

├── requirements.txt

├── Dockerfile

├── app.py

├── plant_disease_model.keras

└── README.md

# 🚀 Running the Project Locally

1️⃣ Clone the repository
git clone https://github.com/your-username/plant-disease-detection.git

2️⃣ Move into the project folder
cd plant-disease-detection

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the application
python app.py

---

# 🌍 Deployment

The application is deployed using Hugging Face Spaces.

# 🔗 Live App:
https://huggingface.co/spaces/ramji-4412/plant-disease-detection

📸 Example Workflow

1️⃣ Upload plant leaf image

2️⃣ Model processes the image

3️⃣ Disease prediction is displayed

---

# 🎯 Future Improvements

Add more plant disease classes

Improve model accuracy

Provide treatment suggestions for diseases

Create a mobile app version

Add real-time camera detection

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve the project:

Fork the repository

Create a new branch

Submit a pull request

---

# 📜 License

This project is for educational and research purposes.

---

# 👨‍💻Team Memebers

1. Raghav Pathak (Lead)

2. Rahul Yadav

3. Ram

4. Ram Ji Trivedi
