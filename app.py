from flask import Flask, request, render_template, jsonify # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
from deepface import DeepFace # type: ignore

app = Flask(__name__)

def detect_faces(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = DeepFace.detectFace(image_rgb, detector_backend='opencv', enforce_detection=False)
    return face_locations

def encode_faces(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    embeddings = DeepFace.represent(image_rgb, model_name='Facenet', enforce_detection=False)
    return embeddings

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    try:
        image = Image.open(file.stream)
        image = np.array(image)

        print("Image uploaded successfully.")
        
        # Detect faces
        face_locations = detect_faces(image)
        print(f"Detected face locations: {face_locations}")

        # Encode faces
        face_encodings = encode_faces(image)
        print(f"Face encodings: {face_encodings}")

        results = []
        for face_encoding in face_encodings:
            results.append({'embedding': face_encoding})
        
        return render_template('index.html', results=results)

    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred", 500

if __name__ == '__main__':
    app.run(debug=True)
