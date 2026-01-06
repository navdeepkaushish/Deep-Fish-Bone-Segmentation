import torch
import cv2 as cv
import numpy as np
from models.f_model import f_UNet # full body seg model
from models.v_model import UNet  # bone seg model
from utils.preprocess import rescale_pad
from utils.postprocess import up_mask, crop_pts, cr_up, v_extract_polygons, draw_polygons
#from utils.visualize import draw_polygons

import mlflow
import time

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fish_inference")

class VentralPipeline:
    def __init__(self, device):
        self.device = device
        self.threshold = 0.45
        self.N = 13
        # full head segmentation
        self.full_model = f_UNet(3, 1)
        ckpt = torch.load("checkpoints/ventral/full_seg.pt", map_location=device)
        pretrained_dict = ckpt['state_dict']
        #pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()} #for distributed model loading
        self.full_model.load_state_dict(pretrained_dict)
        self.full_model.eval().to(device)

        # bone segmentation (13 masks)
        self.bone_model = UNet(3, self.N)
        ckpt = torch.load("checkpoints/ventral/bone_seg.pt", map_location=device)
        pretrained_dict = ckpt['state_dict']
        #pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()} #for distributed model loading
        self.bone_model.load_state_dict(pretrained_dict)
        self.bone_model.eval().to(device)

    def predict_single(self, img):

        start = time.time() # for mlflow tracking

        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
        H, W = img.shape[:2]

        # -------- FULL SEG --------
        re_img, _ = rescale_pad(img, None, 512)
        image = re_img/255.0
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        image = torch.unsqueeze(image, 0)
        image = image.type(torch.float32)
        image = image.to(self.device)
        out = self.full_model(image)
        out = torch.squeeze(out)
        #out = out.cpu().detach().numpy()
        pred = torch.sigmoid(out)
        pred = pred.cpu()
        pred = pred.detach().numpy()
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        u_mask = up_mask(pred, H, W)
        u_mask = u_mask.astype(np.uint8)
        smooth = cv.blur(u_mask, (10, 10))
        pts = crop_pts(smooth) #extract coordinates for cropping
     #====== Bone Seg =====================================
        cr_img = img[pts[1]:pts[3],pts[0]:pts[2]]
        h, w = cr_img.shape[:2]
        re_img, _ = rescale_pad(cr_img, None, 512)
        image = re_img/255.0
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        image = torch.unsqueeze(image, 0)
        image = image.type(torch.float32)
        image = image.to(self.device)

        pred = self.bone_model(image)
        pred = torch.squeeze(pred)
        pred = pred.permute(1,2,0)
        r_logits = pred.cpu().detach().numpy()
        pred = torch.sigmoid(pred)
        pred = pred.cpu()
        pred = pred.detach().numpy()
        pred[pred < self.threshold] = 0
        pred[pred >= self.threshold] = 255
     
        u_mask = up_mask(pred, h, w)
        u_mask = u_mask.astype(np.uint8)
        smooth = cv.GaussianBlur(u_mask, (5, 5), 0)
     
        out = cr_up(smooth, pts, H, W)

        polygons = v_extract_polygons(out, self.N, H, W)
        vis = draw_polygons(img.copy(), polygons)

        latency = time.time() - start

        with mlflow.start_run(run_name="ventral_single", nested=True):
            mlflow.log_param("view", "ventral")
            mlflow.log_param("mode", "single")
            mlflow.log_metric("latency_sec", latency)
            mlflow.log_metric("num_polygons", len(polygons))

        return polygons, vis

    def predict_batch(self, images):

        start = time.time()

        results = []
        for img in images:
            results.append(self.predict_single(img))

        latency = time.time() - start

        with mlflow.start_run(run_name="ventral_batch", nested=True):
            mlflow.log_param("view", "ventral")
            mlflow.log_param("mode", "batch")
            mlflow.log_metric("num_images", len(images))
            mlflow.log_metric("total_latency_sec", latency)
            mlflow.log_metric("avg_latency_sec", latency / len(images))



        return results
