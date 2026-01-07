# Zebrafish Bone Structure Segmentation  
**Ventral & Lateral Views | Production-Ready Deep Learning Pipeline**

This repository contains a **production-ready computer vision pipeline** for zebrafish head and bone structure segmentation using **U-Net architectures** implemented in **PyTorch**.  
The system supports **both ventral and lateral views**, **CPU/GPU inference**, **CLI and API usage**, and is fully **Dockerized with CI/CD support**.

---

## Project Overview

Zebrafish images are processed through a multi-stage deep learning pipeline:

### Ventral View
- **Full Head Segmentation**  
   - Binary segmentation to isolate the zebrafish head from the background
- **Head Cropping**  
   - Bounding box extraction from the head mask
- **Bone Structure Segmentation**  
   - Multi-mask segmentation of internal developing bone structures

### Lateral View
- Single segmentation model that predicts a **combined bone mask** from the lateral view
  
<p align="center"> <img src="data/sample_anno/dataset_sample.png" width="700"><br> <em>Figure 1: Dataset Sample and Annotations (a) Lateral and (b) Ventral view</em> </p> <br>


---

## Key Features

- Ventral & lateral view support
- Multi-stage inference pipeline
- CPU / GPU automatic selection
- CLI (offline / batch inference)
- FastAPI (online inference)
- Dockerized for production
- GitHub Actions CI/CD
- MLflow-ready experiment tracking

## Repository Structure

```text
Deep-Fish-Bone-Segmentation/
│
├── app.py                  # CLI entry point (offline inference)
├── serve.py                # FastAPI service (online inference)
├── Dockerfile
├── requirements.txt
├── README.md
├── .dockerignore
│
├── examples/              
│
├── checkpoints/     # Downloaded at runtime or build time (for docker build)
│   |-- v_full_seg.pt
│   │-- v_bone_seg.pt
│   |-- l_bone_seg
│     
│
├── models/
│   |── unet.py
|   |-- download_models.py
│
├── utils/
│   ├── preprocessing.py
│   ├── postprocessing.py
│  
│
├── ventral/
│   |-- inference.py
│   |-- pipeline.py
|
├── lateral/
│   |-- inference.py
│   |-- pipeline.py
|
├── data/                   # Runtime input (gitignored)
├── outputs/                # Runtime outputs (gitignored)
├── mlruns/                 # MLflow logs (gitignored)
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Running the Project

### Local Setup (CPU or GPU)

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\\Scripts\\activate       # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### CLI Inference (Offline)

Uses demo images by default (`examples/`):

```bash
python app.py --view ventral
python app.py --view lateral
```

You can also provide your own images:

```bash
python app.py --view ventral --input /path/to/images
```

Results are saved to `outputs/`.

---

### API Inference (Online)

Start the FastAPI server:

```bash
uvicorn serve:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:
- `/predict` – single image inference
- `/predict_batch` – batch inference

---

## Docker Usage

### Build Docker Image

```bash
docker build -t deep-fish-bone-segmentation .
```

### Run API (GPU if available)

```bash
docker run --gpus all -p 8000:8000 fish-seg
```

### Run CLI in Docker

```bash
docker run --gpus all fish-seg python app.py --view ventral
```

CPU-only systems work automatically without `--gpus`.

---

## Device Handling

The project automatically selects the device:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- GPU is used if available
- CPU fallback otherwise
- Same Docker image works everywhere

---

## CI/CD

GitHub Actions automatically:
- Builds the Docker image
- Runs basic validation on each push / pull request

Workflow file:
```text
.github/workflows/ci.yml
```

---

## Tech Stack

- **PyTorch**
- **U-Net**
- **OpenCV**
- **FastAPI**
- **Docker**
- **MLflow**
- **GitHub Actions**
- **Hugging Face**

---


## License

Apache 2.0.

---

## Author

**Navdeep Kumar**  
PhD in computer Science, specializes in compter vision and deep learning 
Focus: Computer Vision, Deep Learning, Production ML Systems

