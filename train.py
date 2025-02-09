import os
import cv2
import numpy as np
import face_recognition
import pickle
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load images and labels from a dataset folder with subfolders for each class
def load_images_and_labels(dataset_path, img_size=(50, 50)):
    faces = []
    labels = []
    label_names = os.listdir(dataset_path)  # Get the names of subfolders (classes)
    
    for label_name in label_names:
        label_path = os.path.join(dataset_path, label_name)  # Path to the class folder
        if os.path.isdir(label_path):
            logging.info(f"Processing folder: {label_name}")
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                image = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    faces.append(face_encodings[0])
                    labels.append(label_name)  # Label with the subfolder name
                logging.info(f"Processed image: {img_name}")
    
    return np.array(faces), np.array(labels)

# Load the dataset
dataset_path = "static/faces"  # Change this to your dataset path
logging.info("Loading dataset...")
faces, labels = load_images_and_labels(dataset_path)

# Encode the labels to integer values
logging.info("Encoding labels...")
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
logging.info("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(faces, encoded_labels, test_size=0.2, random_state=42)

# Train a K-Nearest Neighbors (KNN) classifier
logging.info("Training KNN classifier...")
knn = KNeighborsClassifier(n_neighbors=5)  # Adjust the number of neighbors if needed
knn.fit(X_train, y_train)

# Test the model's accuracy
logging.info("Testing model accuracy...")
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model accuracy: {accuracy}")

# Save the trained model, label encoder, and face encodings in a single file using pickle
logging.info("Saving trained model and label encoder...")
combined_model = {
    "model": knn,
    "label_encoder": label_encoder,
    "face_encodings": faces,
    "labels": labels
}

with open("static/face_recognition.pkl", "wb") as f:
    pickle.dump(combined_model, f)

logging.info("Model, label encoder, and face encodings have been successfully saved to static/face_recognition.pkl.")
