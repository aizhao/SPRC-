"""Test caption generation on a single image"""
import sys
import torch
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent / "src"))
from lavis.models import load_model_and_preprocess

# Load model
print("Loading BLIP2 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", 
    model_type="pretrain_opt2.7b", 
    is_eval=True, 
    device=device
)

# Test on first image
cirr_path = Path("./cirr_dataset")
import json
with open(cirr_path / "cirr" / "image_splits" / "split.rc2.train.json", 'r') as f:
    name_to_relpath = json.load(f)

# Get first image
first_name = list(name_to_relpath.keys())[0]
img_path = cirr_path / name_to_relpath[first_name]

print(f"\nTesting on: {first_name}")
print(f"Image path: {img_path}")
print(f"Exists: {img_path.exists()}")

if img_path.exists():
    raw_image = Image.open(img_path).convert('RGB')
    print(f"Image size: {raw_image.size}")
    
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    print(f"Processed tensor shape: {image.shape}")
    
    # Try generation
    print("\nGenerating caption...")
    with torch.no_grad():
        try:
            samples = {"image": image}
            caption = model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=3,
                max_length=50,
                min_length=10
            )
            print(f"Result type: {type(caption)}")
            print(f"Result: {caption}")
            if isinstance(caption, list):
                print(f"First caption: '{caption[0]}'")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
