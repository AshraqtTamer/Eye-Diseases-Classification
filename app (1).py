import gradio as gr
import numpy as np
import time
from PIL import Image
import tensorflow as tf
import random
import os

MODEL_PATH = "eye_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

def predict(image):
    if image is None:
        return {"No image uploaded": 1.0}, 0.0
    start = time.time()
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)[0]
    end = time.time()
    results = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    return results, round(end - start, 3)

TEST_DIR = "test_images"

def predict_random_test_image():
    if not os.path.exists(TEST_DIR):
        return None, {"Error": 1.0}, 0.0, "test_images folder missing"
    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not files:
        return None, {"Error": 1.0}, 0.0, "No images found"
    random_img = random.choice(files)
    full_path = os.path.join(TEST_DIR, random_img)
    img = Image.open(full_path)
    label, t = predict(img)
    return img, label, t, full_path

title = "Eye Disease Classification 👁️"
description = "Classifies eye images into Cataract, Diabetic Retinopathy, Glaucoma, or Normal."

with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    with gr.Tab("Upload Image"):
        img_input = gr.Image(type="pil", label="Upload Eye Image")
        output_label = gr.Label(num_top_classes=4)
        output_time = gr.Number(label="Prediction Time (s)")
        btn1 = gr.Button("Predict")
        btn1.click(fn=predict, inputs=img_input, outputs=[output_label, output_time])
    with gr.Tab("Random Test Image"):
        img_output = gr.Image(label="Random Test Image")
        pred_label = gr.Label(num_top_classes=4)
        pred_time = gr.Number(label="Prediction Time (s)")
        img_path = gr.Textbox(label="Image Path")
        btn2 = gr.Button("Test Random Image")
        btn2.click(fn=predict_random_test_image, inputs=None,
                   outputs=[img_output, pred_label, pred_time, img_path])
demo.launch(share=True)
