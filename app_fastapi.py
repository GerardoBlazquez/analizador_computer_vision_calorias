# app_fastapi.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import io, os
from PIL import Image
import numpy as np

# utils
from utils import load_model, load_nutrition_csv, predict_with_nutrition

# CONFIG - ajusta rutas si hace falta
MODEL_PATH = "model/best_efficientnet_b0.pth"
CLASSES_PATH = "models/classes.txt"
NUT_CSV = "data/nutrition_food101_merged.csv"
IMG_SIZE = 192
DEVICE = "cpu"

# Load classes
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"classes.txt no encontrado en {CLASSES_PATH}")
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = [l.strip() for l in f if l.strip()]

# Load model
model = load_model(MODEL_PATH, num_classes=len(classes), device=DEVICE)

# Load nutrition CSV
nut_df, nut_names = load_nutrition_csv(NUT_CSV)

# Pydantic response model
class PredictResp(BaseModel):
    top1_label: str
    top1_prob: float
    top3: list
    nutrition: dict = None
    note: str = ""

app = FastAPI(title="Food Calorie Estimator API")

@app.get("/")
def root():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH), "nutrition_csv": os.path.basename(NUT_CSV)}

@app.post("/predict", response_model=PredictResp)
async def predict(file: UploadFile = File(...), portion_multiplier: float = 1.0, topk: int = 3):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    res = predict_with_nutrition(
        model=model,
        classes=classes,
        nut_df=nut_df,
        nut_names=nut_names,
        image=img,
        image_bytes=None,
        topk=topk,
        portion_multiplier=portion_multiplier,
        device=DEVICE,
        img_size=IMG_SIZE
    )

    return PredictResp(
        top1_label=res["top1_label"],
        top1_prob=res["top1_prob"],
        top3=res["topk"],
        nutrition=res["nutrition"],
        note=res["note"]
    )
