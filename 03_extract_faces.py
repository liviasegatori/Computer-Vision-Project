import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse

# --- CONFIGURATION ---
# Name of the pretrained model to use for facial expression recognition
MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"

# Fallback mapping in case the model's config labels are generic (e.g., "LABEL_0")
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
    # Try to resolve a path from the CSV by joining with the manifest directory
    path_attempt_1 = os.path.join(base_dir, relative_path_from_csv)
    if os.path.exists(path_attempt_1):
        return path_attempt_1
    
    # If direct join failed, search recursively in the base directory for the filename
    filename = os.path.basename(relative_path_from_csv)
    candidates = glob.glob(os.path.join(base_dir, "**", filename), recursive=True)
    if candidates:
        return candidates[0]
    return None

def main():
    # Parse command line arguments: input manifest CSV and output parquet path
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Select device: prefer MPS (Apple Silicon) if available, otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on: {device}")

    # Load the pretrained image processor and classification model
    print(f"Loading Model: {MODEL_NAME}...")
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
    except Exception as e:
        # If model download/load fails, print the error and exit
        print(f"Error loading model: {e}")
        return

    # Inspect model label mapping (id2label). Sometimes models provide generic labels like "LABEL_0".
    id2label = model.config.id2label
    first_label = list(id2label.values())[0]
    
    # If the model returns "LABEL_0", use our manual mapping instead of the model's labels
    if "LABEL" in first_label.upper():
        print("Model missing internal labels. Using MANUAL map.")
        id2label = MANUAL_LABELS
    else:
        print(f"Model has internal labels: {id2label}")

    # Check manifest exists
    if not os.path.exists(args.manifest):
        print(f"Manifest not found: {args.manifest}")
        return

    # Read frames info from CSV and prepare output collection
    df = pd.read_csv(args.manifest)
    base_dir = os.path.dirname(args.manifest)
    results = []
    
    print(f"Processing {len(df)} frames...")

    # Iterate over each row/frame in the manifest
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Resolve the image path using the helper that handles relative paths and recursive search
        img_path = find_image_path(base_dir, str(row.get('frame_path', '')))
        
        # Skip if image not found
        if not img_path: continue

        # Prepare the basic result structure with video/time metadata
        row_result = {
            "video_id": row['video_id'],
            "t_start": row['t_start'],
            "t_end": row['t_end']
        }

        try:
            # Open image and convert to RGB (model expects 3-channel images)
            image = Image.open(img_path).convert("RGB")
            # Preprocess the image into model inputs (tensors)
            inputs = processor(images=image, return_tensors="pt")
            # Move tensors to the selected device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Run inference in no-grad mode to get logits and convert to probabilities
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

            # Map scores using the corrected id2label dictionary (lowercased names)
            scores = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
            
            # Assign each expected facial expression column explicitly
            row_result["face_angry"]    = float(probs[0])
            row_result["face_disgust"]  = float(probs[1])
            row_result["face_fear"]     = float(probs[2])
            row_result["face_happy"]    = float(probs[3])
            row_result["face_sad"]      = float(probs[4])
            row_result["face_surprise"] = float(probs[5])
            row_result["face_neutral"]  = float(probs[6])

            # Append processed row to results
            results.append(row_result)

        except Exception:
            # On any processing error for a single image, skip it and continue
            continue

    # If we have any results, write them out as a parquet file
    if results:
        df_out = pd.DataFrame(results)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df_out.to_parquet(args.output)
        print(f"Saved {len(results)} rows to: {args.output}")
    else:
        print("No faces processed.")

if __name__ == "__main__":
    main()