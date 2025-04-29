import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from scipy import stats  # Import for mode calculation

# Set up paths and load the model
model_path = 'models/FairplayAI.h5'
model = load_model(model_path)

# Streamlit UI
st.title("Fairplay AI Image Classifier")
st.write("Upload an image or video to see the model's prediction.")

# Function to preprocess images
def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = image / 255.0  # Normalize the image
    return np.expand_dims(image, axis=0)

# Image and video uploader
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi"])

# Check if a file has been uploaded
if uploaded_file is not None:
    if uploaded_file.type.startswith('video'):
        # Process video
        st.video(uploaded_file)  # Display the video
        video_capture = cv2.VideoCapture(uploaded_file.name)  # Open video file
        
        predictions = []  # To store predictions for each frame
        
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            
            processed_frame = preprocess_image(frame)
            yhat = model.predict(processed_frame)  # Get the probability score
            predicted_class = "Class 1" if yhat > 0.5 else "Class 0"
            predictions.append(predicted_class)  # Store the prediction
            
            # Optional: Display predictions on the frame
            cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            st.image(frame, channels="BGR")  # Display frame with predictions
        
        video_capture.release()

        # Aggregate predictions if not empty
        if predictions:
            final_prediction = stats.mode(predictions).mode[0]  # Get the most common prediction
            st.write(f"Final Prediction for Video: {final_prediction}")

            # Print detailed interpretation
            if final_prediction == "Class 1":
                st.write("This is a clean tackle.")
            else:
                st.write("This is a foul.")
        else:
            st.write("No frames were processed.")
        
    else:
        # Process image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess and make prediction
        processed_image = preprocess_image(np.array(image))
        yhat = model.predict(processed_image)  # Get the probability score
        predicted_class = "Class 1" if yhat > 0.5 else "Class 0"
        
        # Display yhat and class prediction
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence Score (yhat): {yhat[0][0]:.4f}")

        # Print detailed interpretation
        if predicted_class == "Class 1":
            st.write("This is a clean tackle.")
        else:
            st.write("This is a foul.")
