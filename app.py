import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Class labels for tumor classification
CLASS_LABELS = ['meningioma', 'glioma', 'pituitary', 'notumor']

# Load pre-trained segmentation and classification models
@st.cache_resource
def load_segmentation_model():
    return tf.keras.models.load_model("tumor_segmentation.h5")

@st.cache_resource
def load_classification_model():
    return tf.keras.models.load_model("tumor_detection.h5")

segmentation_model = load_segmentation_model()
classification_model = load_classification_model()

# App title and description
st.title("Brain Tumor Detection and Segmentation App")
st.write("Upload an MRI image of the brain to detect the type of tumor and visualize the segmented tumor region.")

# File upload
uploaded_file = st.file_uploader("Upload a brain MRI image (PNG/JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for segmentation
    image_array = np.array(image)
    resized_image = cv2.resize(image_array, (128, 128))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel
    input_image_seg = np.expand_dims(rgb_image / 255.0, axis=0)  # Normalize and add batch dimension

    # Perform segmentation
    segmentation_prediction = segmentation_model.predict(input_image_seg)[0].squeeze()
    segmentation_resized = cv2.resize(segmentation_prediction, (image_array.shape[1], image_array.shape[0]))

    # Preprocess the image for classification
    resized_image_class = cv2.resize(image_array, (224, 224))
    rgb_image_class = cv2.cvtColor(resized_image_class, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel
    input_image_class = np.expand_dims(rgb_image_class / 255.0, axis=0)

    # Perform classification
    classification_prediction = classification_model.predict(input_image_class)
    predicted_class_index = np.argmax(classification_prediction)
    predicted_class_label = CLASS_LABELS[predicted_class_index]
    confidence_score = classification_prediction[0][predicted_class_index]

    # Display results
    st.subheader("Results:")
    st.write(f"**Tumor Type:** {predicted_class_label} ({confidence_score:.2%} confidence)")

    st.subheader("Tumor Segmentation:")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_array, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(segmentation_resized, cmap='gray')
    ax[1].set_title("Segmented Tumor Mask")
    ax[1].axis("off")
    st.pyplot(fig)
