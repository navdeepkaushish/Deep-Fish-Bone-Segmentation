import torch
import cv2 as cv
import numpy as np
from models.l_model import UNet
from utils.preprocess import rescale_pad
from utils.postprocess import up_mask, crop_pts, cr_up, l_extract_polygons
import mlflow
import time
#from utils.visualize import draw_polygons

mlflow.set_tracking_uri("file:./mlruns")  # or your store_uri

mlflow.set_experiment("Deep-Fish-Bone-Segmentation")

class LateralPipeline:
    def __init__(self, device):
        self.device = device
        self.threshold = 0.45

        self.model = UNet(3, 1)
        ckpt = torch.load("checkpoints/lateral/bone_seg.pt", map_location=device)
        pretrained_dict = ckpt['state_dict']
        pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()} #for distributed model loading
        self.model.load_state_dict(pretrained_dict)
        self.model.eval().to(device)

    def predict_single(self, img):

        start = time.time() # for mlflow tracking

        H, W = img.shape[:2]
        re_img, _ = rescale_pad(img, None, 512)

        x = torch.from_numpy(re_img/255.).permute(2,0,1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            out = torch.sigmoid(self.model(x))[0,0].cpu().numpy()

        mask = (out >= self.threshold).astype(np.uint8)
        mask = up_mask(mask, H, W)

        polygons = l_extract_polygons(mask)
        img_cp = img.copy()
        cv.drawContours(img_cp, polygons, -1, (255, 0, 0), 2, cv.LINE_AA)

        latency = time.time() - start

        with mlflow.start_run(run_name="lateral_single", nested=True):
            mlflow.log_param("view", "lateral")
            mlflow.log_param("mode", "single")
            mlflow.log_metric("latency_sec", latency)
            mlflow.log_metric("num_polygons", len(polygons))

        return polygons, img_cp

    def predict_batch(self, images):
        start = time.time()

        results = []
        for img in images:
            results.append(self.predict_single(img))

        latency = time.time() - start

        with mlflow.start_run(run_name="lateral_batch", nested=True):
            mlflow.log_param("view", "lateral")
            mlflow.log_param("mode", "batch")
            mlflow.log_metric("num_images", len(images))
            mlflow.log_metric("total_latency_sec", latency)
            mlflow.log_metric("avg_latency_sec", latency / len(images))


        return results
