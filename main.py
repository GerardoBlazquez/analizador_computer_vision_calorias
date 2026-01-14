# app.py
import gradio as gr
from PIL import Image
import torch
import os

from utils import load_model, load_nutrition_csv, predict_with_nutrition

# CONFIG
MODEL_PATH = "model/best_efficientnet_b0.pth"
CLASSES_PATH = "models/classes.txt"
NUT_CSV = "data/nutrition_food101_merged.csv"
IMG_SIZE = 192
DEVICE = "cpu"

# Load classes
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = [l.strip() for l in f if l.strip()]

# Load model & nutrition
model = load_model(MODEL_PATH, num_classes=len(classes), device=DEVICE)
nut_df, nut_names = load_nutrition_csv(NUT_CSV)

def infer(image: Image.Image, portion: float):
    res = predict_with_nutrition(
        model=model,
        classes=classes,
        nut_df=nut_df,
        nut_names=nut_names,
        image=image,
        image_bytes=None,
        topk=3,
        portion_multiplier=portion,
        device=DEVICE,
        img_size=IMG_SIZE
    )

    return {
        "Predicción": res["top1_label"],
        "Confianza": res["top1_prob"],
        "Top-3": res["topk"],
        "Calorías estimadas": res["nutrition"]["calories_per_serving"] if res["nutrition"] else None,
        "Macronutrientes": res["nutrition"]
    }

iface = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(type="pil", label="Sube una imagen de comida"),
        gr.Slider(0.25, 3.0, value=1.0, step=0.25, label="Tamaño de la porción")
    ],
    outputs="json",
    title="🍕 Food Calorie Estimator",
    description="Reconocimiento de comida + estimación nutricional (Food-101)"
)

iface.launch(share=True)
