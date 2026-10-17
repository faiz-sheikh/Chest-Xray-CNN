import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

HF_REPO_ID = "faiz4320/cnn-model"
HF_MODEL_FILENAME = "model.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "INVALID",
    "NORMAL",
    "PNEUMONIA"
]


# ------------------------------------------------------------
# Download model from Hugging Face and load it
# ------------------------------------------------------------

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME
    )

    model = tf.keras.models.load_model(
        model_path
    )

    return model


# ------------------------------------------------------------
# Streamlit page
# ------------------------------------------------------------

st.set_page_config(
    page_title="Chest X-Ray Classification",
    page_icon="🩻",
    layout="centered"
)

st.title(
    "Chest X-Ray Classification"
)

st.write(
    "Upload an image to classify it as "
    "Invalid, Normal, or Pneumonia."
)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

try:

    model = load_model()

except Exception as error:

    st.error(
        "Unable to load the model from Hugging Face."
    )

    st.exception(error)

    st.stop()


# ------------------------------------------------------------
# Image upload
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ]
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )


    # Resize exactly as during training
    image = image.resize(
        IMAGE_SIZE
    )


    image_array = np.array(
        image,
        dtype=np.float32
    )


    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # Output order:
    #
    # [INVALID, NORMAL, PNEUMONIA]

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]


    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    confidence = float(
        probabilities[
            predicted_index
        ]
    )


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    st.subheader(
        "Prediction Result"
    )


    if predicted_class == "INVALID":

        st.warning(
            "Prediction: INVALID / UNSUPPORTED IMAGE"
        )

        st.write(
            "Please upload a valid chest X-ray image."
        )


    elif predicted_class == "NORMAL":

        st.success(
            "Prediction: NORMAL"
        )


    elif predicted_class == "PNEUMONIA":

        st.error(
            "Prediction: PNEUMONIA"
        )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    st.write(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )


    st.write(
        f"Invalid score: "
        f"{probabilities[0] * 100:.2f}%"
    )


    st.write(
        f"Normal score: "
        f"{probabilities[1] * 100:.2f}%"
    )


    st.write(
        f"Pneumonia score: "
        f"{probabilities[2] * 100:.2f}%"
    )


    st.warning(
        "Educational project only. "
        "Not suitable for medical diagnosis."
    )
