"""
Generate captions for reference images in CIRR dataset using BLIP2
This enhances the modification text with reference image descriptions
"""
import sys
import torch
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from lavis.models import load_model_and_preprocess

def generate_cirr_captions(split='train'):
    """Generate captions for CIRR reference images"""
    
    # Load BLIP2 model for caption generation
    print("Loading BLIP2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", 
        model_type="pretrain_opt2.7b", 
        is_eval=True, 
        device=device
    )
    
    # Load CIRR annotations
    cirr_path = Path("./cirr_dataset")
    annotation_file = cirr_path / "cirr" / "captions" / f"cap.rc2.{split}.json"
    
    print(f"Loading {split} annotations from {annotation_file}")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Load image path mapping
    with open(cirr_path / "cirr" / "image_splits" / f"split.rc2.{split}.json", 'r') as f:
        name_to_relpath = json.load(f)
    
    # Generate captions
    captions_dict = {}
    image_base_dir = cirr_path
    
    print(f"Generating captions for {len(annotations)} samples...")
    success_count = 0
    fail_count = 0
    
    for item in tqdm(annotations):
        reference_name = item['reference']
        
        # Skip if already generated
        if reference_name in captions_dict:
            continue
        
        # Load reference image using relative path
        if reference_name not in name_to_relpath:
            captions_dict[reference_name] = ""
            fail_count += 1
            continue
        
        img_path = image_base_dir / name_to_relpath[reference_name]
        if not img_path.exists():
            captions_dict[reference_name] = ""
            fail_count += 1
            continue
        
        try:
            raw_image = Image.open(img_path).convert('RGB')
            # Check if image is valid
            if raw_image.size[0] == 0 or raw_image.size[1] == 0:
                captions_dict[reference_name] = ""
                fail_count += 1
                continue
                
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            
            # Generate caption using BLIP2 with correct parameters
            with torch.no_grad():
                samples = {"image": image}
                # Use larger max_length to avoid negative length issue
                captions = model.generate(
                    samples, 
                    use_nucleus_sampling=False,
                    num_beams=3,
                    max_length=50,
                    min_length=10
                )
                caption = captions[0] if isinstance(captions, list) else captions
                
                if caption and len(caption.strip()) > 0:
                    captions_dict[reference_name] = caption
                    success_count += 1
                else:
                    captions_dict[reference_name] = ""
                    fail_count += 1
            
        except Exception as e:
            captions_dict[reference_name] = ""
            fail_count += 1
    
    print(f"\nSuccess: {success_count}, Failed: {fail_count}")
    
    # Save captions
    output_file = cirr_path / "cirr" / "captions" / f"reference_captions.{split}.json"
    print(f"Saving captions to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(captions_dict, f, indent=2)
    
    print(f"Generated {len(captions_dict)} captions")
    return captions_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test1"], 
                        help="Dataset split")
    args = parser.parse_args()
    
    generate_cirr_captions(args.split)
