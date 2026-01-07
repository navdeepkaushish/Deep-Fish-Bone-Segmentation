import os
from huggingface_hub import hf_hub_download

os.makedirs("checkpoints", exist_ok=True)

models = {
    "v_bone_seg.pt": "navdeepkaushish/fish-bone-segmentation-models",
    "l_bone_seg.pt": "navdeepkaushish/fish-bone-segmentation-models",
    "v_full_seg.pt": "navdeepkaushish/fish-bone-segmentation-models"
}

for name, repo_id in models.items():
    output_path = os.path.join("checkpoints", name)
    if not os.path.exists(output_path):
        print(f"Downloading {name}...")
        hf_hub_download(repo_id=repo_id, filename=name, cache_dir="checkpoints")
    else:
        print(f"{name} already exists, skipping.")
