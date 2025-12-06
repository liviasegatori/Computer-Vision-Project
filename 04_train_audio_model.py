"""
Audio Model Training (Transfer Learning)
----------------------------------------
This script fine-tunes a pre-trained Wav2Vec2 model on the IEMOCAP dataset.

Methodology:
1. Domain Adaptation: Takes a generic speech model (Wav2Vec2-Base) and trains it
   to recognize emotional valence/arousal.
2. Optimization: Uses Gradient Accumulation to simulate large batch sizes on
   consumer hardware (essential for Transformer training).
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ====== IMPORT ======
import os, re, glob, json
import numpy as np
import torch
from datasets import Dataset, DatasetDict, Features, ClassLabel, Audio
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
import soundfile as sf

# ====== CONFIGURATION ======
DATA_DIR = "Session1"  
OUTPUT_DIR = "./wav2vec2-iemocap-local"
MODEL_NAME = "facebook/wav2vec2-base"
SEED = 42

# ====== HARDWARE DETECTION ======
if torch.cuda.is_available():
    device = "cuda"
    use_fp16 = True
    print("Hardware: NVIDIA CUDA GPU detected.")
elif torch.backends.mps.is_available():
    device = "mps"
    use_fp16 = False # MPS implies Apple Silicon, which doesn't use the 'fp16' flag typically
    print("Hardware: Apple Silicon (MPS) detected.")
else:
    device = "cpu"
    use_fp16 = False
    print("Hardware: CPU only (Training will be slow).")

# ====== IEMOCAP PARSING ======
if not os.path.isdir(DATA_DIR):
    print(f"Folder '{DATA_DIR}' does not exist in directory.")
    raise RuntimeError(f"DATA_DIR not found: {DATA_DIR}")

EMOTIONS = {"ang","hap","exc","neu","sad","fru","sur","fea","dis","oth","xxx"}
utt_pat = re.compile(r"Ses\d+[MF]_[^\s\]]+")
utt2label = {}

# Locate text files
cat_txts = glob.glob(os.path.join(DATA_DIR, "dialog", "EmoEvaluation", "**", "*.txt"), recursive=True)

# Parse labels
for txt in cat_txts:
    with open(txt, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            m_id = utt_pat.search(line)
            if not m_id:
                continue
            utt = m_id.group(0)
            chunks = re.findall(r"\[([^\]]+)\]", line)
            chunks.append(line)
            label = None
            for ch in chunks:
                for t in re.split(r"[\s,;/]+", ch.lower()):
                    if t in EMOTIONS:
                        label = t
                        break
                if label:
                    break
            if label and label != "xxx":
                utt2label[utt] = label

rows = []
wav_files = glob.glob(os.path.join(DATA_DIR, "sentences", "wav", "**", "*.wav"), recursive=True)
for wav_path in wav_files:
    base = os.path.splitext(os.path.basename(wav_path))[0]
    if base in utt2label:
        rows.append({"audio": wav_path, "label": utt2label[base]})

print(f"Found {len(utt2label)} labels and {len(rows)} labeled audio files.")


features = Features({"audio": Audio(sampling_rate=16000, decode=False),
                     "label": ClassLabel(names=sorted({r['label'] for r in rows}))})
ds = Dataset.from_list(rows).cast(features)

# Split Train/Test
split = ds.train_test_split(test_size=0.2, seed=SEED)
dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

label_names = dataset["train"].features["label"].names
id2label = {i: l for i, l in enumerate(label_names)}
label2id = {l: i for i, l in enumerate(label_names)}

# ====== PROCESSOR & MODEL ======
print(f"Loading model: {MODEL_NAME}...")

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME, token=False)

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(id2label),
    label2id=label2id,
    id2label=id2label,
    problem_type="single_label_classification",
    ignore_mismatched_sizes=True,
    token=False 
)

# Move model to device manually if needed, though Trainer handles it mostly
model.gradient_checkpointing_enable()

set_seed(SEED)

# ====== PREPROCESS ======
def preprocess(batch):
    path = batch["audio"]["path"] if isinstance(batch["audio"], dict) else batch["audio"]
    # Force 16k loading
    try:
        audio, sr = sf.read(path, always_2d=False)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return {"input_values": np.zeros(16000)} # Return silent dummy on error

    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
        
    # Cut to 30s max to prevent RAM spikes
    if isinstance(audio, np.ndarray) and sr and audio.shape[0] / sr > 30.0:
        audio = audio[: int(30.0 * sr)]
        
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
        
    inputs = processor(audio, sampling_rate=sr, return_tensors=None)
    # Handle cases where input might be empty
    if "input_values" in inputs and len(inputs["input_values"]) > 0:
        batch["input_values"] = inputs["input_values"][0]
    else:
        batch["input_values"] = np.zeros(16000) # Fallback
        
    return batch

print("Preprocessing dataset (this might take a moment)...")
dataset = dataset.map(preprocess, remove_columns=[c for c in dataset["train"].column_names if c not in ("label","audio")])

# ====== COLLATOR ======
class DataCollatorForAudio:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, features):
        input_values = [f["input_values"] for f in features]
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        
        batch = self.processor.pad({"input_values": input_values}, padding="longest", return_tensors="pt")
        batch["labels"] = labels
        return batch

collator = DataCollatorForAudio(processor)

# ====== TRAINING ARGS (LOCAL OPTIMIZED) ======
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # --- MEMORY OPTIMIZATION FOR LOCAL ---
    per_device_train_batch_size=1,   # Keep extremely low for local VRAM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,   # High accumulation to compensate for low batch size (1*8 = 8 effective)
    dataloader_num_workers=0,        # 0 IS CRITICAL LOCALLY 
    
    # --- GENERAL SETTINGS ---
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    weight_decay=0.01,
    num_train_epochs=3,              # Reduced to 3 for a quicker local test run
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    group_by_length=True,
    report_to="none",
    
    # --- HARDWARE SPECIFIC ---
    fp16=use_fp16,                   # Only True if NVIDIA GPU
    use_mps_device=(device == "mps") # Explicitly enable MPS if on Mac
)

# ====== TRAINER ======
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor,
    data_collator=collator,
)

# ====== RUN ======
print(f">> Starting Training on {device.upper()}...")
trainer.train()

print(">> Final Evaluation...")
stats = trainer.evaluate()
print(stats)

# ====== SAVE ======
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)
print(">> Done. Model saved in:", OUTPUT_DIR)