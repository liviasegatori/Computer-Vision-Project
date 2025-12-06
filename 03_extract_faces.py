import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse

# --- CONFIGURATION ---
MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"

MANUAL_LABELS = {
    0: "face_angry",
    1: "face_disgust",
    2: "face_fear",
    3: "face_happy",
    4: "face_sad",
    5: "face_surprise",
    6: "face_neutral"
}

def find_image_path(base_dir, relative_path_from_csv):
    path_attempt_1 = os.path.join(base_dir, relative_path_from_csv)
    if os.path.exists(path_attempt_1):
        return path_attempt_1
    
    filename = os.path.basename(relative_path_from_csv)
    candidates = glob.glob(os.path.join(base_dir, "**", filename), recursive=True)
    if candidates:
        return candidates[0]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on: {device}")

    print(f"Loading Model: {MODEL_NAME}...")
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    id2label = model.config.id2label
    first_label = list(id2label.values())[0]
    
    # If the model returns "LABEL_0", we use our manual map
    if "LABEL" in first_label.upper():
        print("Model missing internal labels. Using MANUAL map.")
        id2label = MANUAL_LABELS
    else:
        print(f"Model has internal labels: {id2label}")

    if not os.path.exists(args.manifest):
        print(f"Manifest not found: {args.manifest}")
        return

    df = pd.read_csv(args.manifest)
    base_dir = os.path.dirname(args.manifest)
    results = []
    
    print(f"Processing {len(df)} frames...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = find_image_path(base_dir, str(row.get('frame_path', '')))
        
        if not img_path: continue

        row_result = {
            "video_id": row['video_id'],
            "t_start": row['t_start'],
            "t_end": row['t_end']
        }

        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

            # Map scores using the corrected id2label dictionary
            scores = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
            
            # Map to specific columns
            row_result["face_angry"]    = float(probs[0])
            row_result["face_disgust"]  = float(probs[1])
            row_result["face_fear"]     = float(probs[2])
            row_result["face_happy"]    = float(probs[3])
            row_result["face_sad"]      = float(probs[4])
            row_result["face_surprise"] = float(probs[5])
            row_result["face_neutral"]  = float(probs[6])

            results.append(row_result)

        except Exception:
            continue

    if results:
        df_out = pd.DataFrame(results)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df_out.to_parquet(args.output)
        print(f"Saved {len(results)} rows to: {args.output}")
    else:
        print("No faces processed.")

if __name__ == "__main__":
    main()