import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

## ----
def streamlit_image_to_cv2(image):
    image = Image.open(image)
    # Convert the image to a Numpy array and then read it into OpenCV format
    image = np.array(image.convert('RGB'))
    image = image[:, :, ::-1].copy()
    return image

def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_score = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_score

## ----

st.title("Simple Face Anti-Spoofing Application 🤥")
image = st.file_uploader("Upload Image", type=['jpg', 'jpeg'])

if image is not None:
    # model = load_model("model/VGG-16_lr104.h5", custom_objects={"f1_score": f1_score })

    img = streamlit_image_to_cv2(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, channels="RGB")

    processed_img = preprocess_input(img)/255
    processed_img = cv2.resize(processed_img, (224,224))
    processed_img = np.expand_dims(processed_img, 0)

    interpreter_quant = tf.lite.Interpreter(model_path=str("vgg16_tflite_model/vgg16_tflite.tflite"))
    interpreter_quant.allocate_tensors()

    input_index = interpreter_quant.get_input_details()[0]["index"]
    output_index = interpreter_quant.get_output_details()[0]["index"]

    interpreter_quant.set_tensor(input_index, processed_img)
    interpreter_quant.invoke()

    probability = interpreter_quant.get_tensor(output_index)
    label = "spoofed" if probability[0][0] > 0.5  else "real"

    st.subheader('Prediction', divider='rainbow')
    st.text(f"This biometric representation is {label}.")
else:
    st.warning('No files have been uploaded', icon="⚠️")


