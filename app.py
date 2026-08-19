from pathlib import Path

import numpy as np
import streamlit as st
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mood_model.tflite"
IMAGE_SIZE = (200, 200)
CLASS_NAMES = ["Happy", "Not Happy"]


st.set_page_config(page_title="Mood Classifier", page_icon="🙂")


@st.cache_resource
def load_interpreter():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "mood_model.tflite is missing. Run train_model.py and commit the generated model."
        )
    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter


def predict_mood(image: Image.Image) -> float:
    interpreter = load_interpreter()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    array = np.asarray(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32)
    array = np.expand_dims(array / 200.0, axis=0)
    interpreter.set_tensor(input_details["index"], array.astype(input_details["dtype"]))
    interpreter.invoke()
    return float(interpreter.get_tensor(output_details["index"])[0][0])


st.title("🙂 Mood Classification")
st.write("Upload a face image to classify it as **Happy** or **Not Happy**.")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Classify mood", type="primary"):
        try:
            score = predict_mood(image)
            class_index = int(score >= 0.5)
            confidence = score if class_index else 1 - score
            st.success(
                f"Prediction: **{CLASS_NAMES[class_index]}** ({confidence:.1%} confidence)"
            )
            st.progress(confidence)
        except Exception as error:
            st.error(f"Could not classify the image: {error}")
