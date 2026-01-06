import cv2
import glob
import torch
from ventral.pipeline import VentralPipeline

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = VentralPipeline(device)

    images = glob.glob("data/v*.png")

    if len(images) == 1:
        img = cv2.imread(images[0])[:,:,::-1]
        poly, vis = pipe.predict_single(img)
        cv2.imwrite("output/ventral_out.png", vis[:,:,::-1])
    else:
        imgs = [cv2.imread(p)[:,:,::-1] for p in images]
        results = pipe.predict_batch(imgs)
        for i, (_, vis) in enumerate(results):
            cv2.imwrite(f"output/ventral_out_{i+1}.png", vis[:,:,::-1])
