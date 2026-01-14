# app_gradio.py
import os
import gradio as gr
import requests
from PIL import Image
import io

# Opcional: si quieres que Gradio llame al backend remoto, pon BACKEND_URL
# Si lo dejas vacío, intentará usar el modelo local (require utils.py + model + data en el mismo entorno)
BACKEND_URL = os.getenv("BACKEND_URL", "").strip()  # e.g. "http://localhost:8000"

def infer_via_api(pil_img, portion):
    # llama al endpoint /predict
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    buffered.seek(0)
    files = {"file": ("img.jpg", buffered, "image/jpeg")}
    params = {"portion_multiplier": portion, "topk": 3}
    resp = requests.post(f"{BACKEND_URL}/predict", files=files, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

# fallback a inferencia local si no hay BACKEND_URL
def infer_local(pil_img, portion):
    from utils import load_model, load_nutrition_csv, predict_with_nutrition
    MODEL_PATH = "model/best_efficientnet_b0.pth"
    NUT_CSV = "data/nutrition_food101_merged.csv"
    CLASSES_PATH = "models/classes.txt"

    # carga clases
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = [l.strip() for l in f if l.strip()]

    model = load_model(MODEL_PATH, num_classes=len(classes), device="cpu")
    nut_df, nut_names = load_nutrition_csv(NUT_CSV)

    res = predict_with_nutrition(
        model=model,
        classes=classes,
        nut_df=nut_df,
        nut_names=nut_names,
        image=pil_img,
        topk=3,
        portion_multiplier=portion,
        device="cpu"
    )
    # adapt output a formato legible
    return res

def infer(pil_img, portion):
    if BACKEND_URL:
        return infer_via_api(pil_img, portion)
    else:
        return infer_local(pil_img, portion)

iface = gr.Interface(
    fn=infer,
    inputs=[gr.Image(type="pil"), gr.Slider(0.25, 3.0, 1.0, label="Porción (multiplicador)")],
    outputs="json",
    title="Food Calorie Estimator",
    description="Sube una foto de comida. Si no configuras BACKEND_URL, la inferencia se hace localmente."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
