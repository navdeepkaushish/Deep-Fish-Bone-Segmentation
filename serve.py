from fastapi import FastAPI, UploadFile, File, Query
from typing import List
import numpy as np
import torch
from PIL import Image
import io
import cv2
import base64
import time
import mlflow

from ventral.pipeline import VentralPipeline
from lateral.pipeline import LateralPipeline
from utils.postprocess import serialize_polygons

# ---------------- MLflow setup ----------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fish_inference")

# ---------------- App ----------------
app = FastAPI(title="Fish Segmentation Service")

device = "cuda" if torch.cuda.is_available() else "cpu"

ventral_pipe = VentralPipeline(device)
lateral_pipe = LateralPipeline(device)

def encode_image(img):
    _, buffer = cv2.imencode(".png", img[:, :, ::-1])
    return base64.b64encode(buffer).decode("utf-8")

#===== single image endpoint =============================
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    view: str = Query(..., enum=["ventral", "lateral"])
):
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    img = np.array(img)

    start = time.time()

    if view == "ventral":
        polygons, vis = ventral_pipe.predict_single(img)
    else:
        polygons, vis = lateral_pipe.predict_single(img)

    latency = time.time() - start

    # ---- MLflow logging ----
    with mlflow.start_run(run_name=f"{view}_single", nested=True):
        mlflow.log_param("view", view)
        mlflow.log_param("mode", "single")
        mlflow.log_metric("latency_sec", latency)
        mlflow.log_metric("num_polygons", len(polygons))

    return {
        "view": view,
        "polygons": serialize_polygons(polygons),
        "annotated_image": encode_image(vis)
    }

#==== Batch inference endpoint ====================
@app.post("/predict_batch")
async def predict_batch(
    images: List[UploadFile] = File(...),
    view: str = Query(..., enum=["ventral", "lateral"])
):
    imgs = []
    for f in images:
        img = Image.open(io.BytesIO(await f.read())).convert("RGB")
        imgs.append(np.array(img))

    start = time.time()

    if view == "ventral":
        results = ventral_pipe.predict_batch(imgs)
    else:
        results = lateral_pipe.predict_batch(imgs)

    latency = time.time() - start

    # ---- MLflow logging ----
    with mlflow.start_run(run_name=f"{view}_batch", nested=True):
        mlflow.log_param("view", view)
        mlflow.log_param("mode", "batch")
        mlflow.log_metric("num_images", len(imgs))
        mlflow.log_metric("total_latency_sec", latency)
        mlflow.log_metric("avg_latency_sec", latency / len(imgs))

    response = []
    for polygons, vis in results:
        response.append({
            "polygons": serialize_polygons(polygons),
            "annotated_image": encode_image(vis)
        })

    return {
        "view": view,
        "results": response
    }
