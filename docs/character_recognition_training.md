# Character Recognition Training Pipeline

**Goal**: Train a character-aware vision model using LoRA fine-tuning on DAF:re dataset to improve Stage 5 visual matching accuracy.

**Created**: 2025-10-03
**Status**: Planning

---

## Problem Statement

**Current Issue**: Generic OpenCLIP ViT-L/14 doesn't understand anime characters
- Same character in different poses/styles: 0.663 similarity (should be 0.9+)
- Different characters with similar features: 0.7-0.8 similarity (too high)
- Model trained on general images, not anime-specific

**Solution**: Fine-tune OpenCLIP with LoRA on character recognition task using DAF:re dataset

---

## DAF:re Dataset

**Source**: https://github.com/arkel23/animesion
**Paper**: https://arxiv.org/abs/2101.08674
**Size**: ~500,000 character images across 3,000+ characters
**Format**: Images + character labels + anime associations

**Pros**:
- Large-scale character data
- Multi-image per character (different poses, expressions)
- Ready for character classification task

**Cons**:
- Likely 2020-2021 era anime (outdated for recent shows like Dandadan)
- Will need incremental updates with new characters
- May not have full coverage of older anime either

**Mitigation**:
1. Train base model on DAF:re for character recognition skills
2. Incrementally add new characters from Stage 5 enrichment outputs
3. Continuous learning pipeline

---

## Architecture

### Base Model
- **OpenCLIP ViT-L/14** (laion2b_s32b_b82k)
- 768-dimensional embeddings
- Pre-trained on 2B image-text pairs

### LoRA Configuration
```python
{
    "r": 8,              # Low-rank dimension
    "alpha": 32,         # Scaling factor
    "dropout": 0.1,      # Regularization
    "target_modules": [  # Which layers to adapt
        "q_proj",
        "v_proj",
        "k_proj",
        "out_proj"
    ]
}
```

### Training Head
```python
CharacterClassifier
├── Vision Encoder (ViT-L/14 + LoRA)
├── Projection Layer (768 → 512)
├── Dropout (0.1)
└── Classification Head (512 → 3000+ characters)
```

---

## Training Pipeline

### Phase 1: Dataset Preparation (Week 1)

**1.1 Download DAF:re Dataset**
```bash
# Download from GitHub releases
wget https://github.com/arkel23/animesion/releases/download/v1.0/dafre.tar.gz
tar -xzf dafre.tar.gz -C data/dafre/

# Expected structure:
# data/dafre/
# ├── images/
# │   ├── character_001/
# │   │   ├── img_001.jpg
# │   │   ├── img_002.jpg
# │   │   └── ...
# │   └── character_002/
# │       └── ...
# ├── metadata.json
# └── character_mapping.json
```

**1.2 Analyze Dataset Structure**
```python
import json
from pathlib import Path

# Load metadata
with open("data/dafre/metadata.json") as f:
    metadata = json.load(f)

# Analyze distribution
total_images = len(list(Path("data/dafre/images").rglob("*.jpg")))
num_characters = len(list(Path("data/dafre/images").iterdir()))
avg_images_per_char = total_images / num_characters

print(f"Total images: {total_images}")
print(f"Characters: {num_characters}")
print(f"Avg images/character: {avg_images_per_char:.1f}")
```

**1.3 Create Training Splits**
```python
# 80/10/10 split: train/val/test
# Stratified by character (ensure each character in all splits)
from sklearn.model_selection import train_test_split

characters = sorted(Path("data/dafre/images").iterdir())
train_chars, temp_chars = train_test_split(characters, test_size=0.2, random_state=42)
val_chars, test_chars = train_test_split(temp_chars, test_size=0.5, random_state=42)

# Save splits
with open("data/dafre/splits.json", "w") as f:
    json.dump({
        "train": [c.name for c in train_chars],
        "val": [c.name for c in val_chars],
        "test": [c.name for c in test_chars]
    }, f, indent=2)
```

### Phase 2: Training Script (Week 1-2)

**File**: `scripts/train_character_recognition.py`

```python
#!/usr/bin/env python3
"""
Train character recognition model using LoRA fine-tuning on DAF:re dataset.

Usage:
    uv run python scripts/train_character_recognition.py --config configs/character_lora.yaml
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
import open_clip
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
import json
from tqdm import tqdm


class DAFreDataset(Dataset):
    """DAF:re character dataset for training."""

    def __init__(self, data_dir: str, split: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Load splits
        with open(self.data_dir / "splits.json") as f:
            splits = json.load(f)

        # Load character mapping
        with open(self.data_dir / "character_mapping.json") as f:
            self.char_to_idx = json.load(f)

        # Collect image paths and labels
        self.samples = []
        for char_id in splits[split]:
            char_dir = self.data_dir / "images" / char_id
            for img_path in char_dir.glob("*.jpg"):
                self.samples.append((img_path, self.char_to_idx[char_id]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class CharacterRecognitionModel(nn.Module):
    """Character recognition with LoRA-enhanced vision encoder."""

    def __init__(self, num_characters: int, lora_config: Dict):
        super().__init__()

        # Load OpenCLIP base model
        self.vision_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="laion2b_s32b_b82k"
        )

        # Apply LoRA
        lora_cfg = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            target_modules=lora_config["target_modules"]
        )
        self.vision_model = get_peft_model(self.vision_model.visual, lora_cfg)

        # Classification head
        self.projection = nn.Linear(768, 512)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(512, num_characters)

    def forward(self, images):
        # Extract features with LoRA-enhanced vision encoder
        features = self.vision_model(images)

        # Classification
        x = self.projection(features)
        x = torch.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%"
        })

    return total_loss / len(dataloader), 100 * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), 100 * correct / total


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/dafre", help="DAF:re data directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="models/character_lora")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load character mapping to get num_characters
    with open(Path(args.data_dir) / "character_mapping.json") as f:
        char_mapping = json.load(f)
    num_characters = len(char_mapping)
    print(f"Training on {num_characters} characters")

    # Create model
    lora_config = {
        "r": 8,
        "alpha": 32,
        "dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"]
    }
    model = CharacterRecognitionModel(num_characters, lora_config)
    model = model.to(device)

    # Create datasets
    _, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14")
    train_dataset = DAFreDataset(args.data_dir, "train", transform=preprocess)
    val_dataset = DAFreDataset(args.data_dir, "val", transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save LoRA weights only (2-7MB)
            torch.save(model.vision_model.state_dict(), output_path / "lora_weights.pt")
            print(f"Saved best model with val_acc: {val_acc:.2f}%")

    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
```

### Phase 3: Integration with Stage 5 (Week 2)

**3.1 Load LoRA Weights in VisionProcessor**

```python
# In src/vector/processors/vision_processor.py

class VisionProcessor:
    def __init__(self, settings: Optional[Settings] = None, lora_weights_path: Optional[str] = None):
        # ... existing init code ...

        # Load LoRA weights if provided
        if lora_weights_path and Path(lora_weights_path).exists():
            self._load_lora_weights(lora_weights_path)

    def _load_lora_weights(self, weights_path: str):
        """Load LoRA weights for character-aware vision model."""
        try:
            from peft import PeftModel, LoraConfig

            # Configure LoRA
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
            )

            # Apply LoRA to vision model
            self.model["model"].visual = PeftModel.from_pretrained(
                self.model["model"].visual,
                weights_path
            )

            logger.info(f"Loaded LoRA weights from {weights_path}")
        except Exception as e:
            logger.warning(f"Failed to load LoRA weights: {e}")
```

**3.2 Enable in Stage 5**

```python
# In scripts/process_stage5_characters.py or ai_character_matcher.py

# Initialize vision processor with LoRA weights
lora_weights_path = "models/character_lora/lora_weights.pt"
if Path(lora_weights_path).exists():
    vision_processor = VisionProcessor(settings, lora_weights_path=lora_weights_path)
    logger.info("Using character-aware vision model with LoRA fine-tuning")
else:
    vision_processor = VisionProcessor(settings)
    logger.info("Using base OpenCLIP model (no LoRA)")
```

---

## Expected Results

### Before LoRA Fine-tuning (Current)
- Same character, different images: **0.66 similarity** ❌
- Different characters: **0.70-0.82 similarity** ⚠️
- False positive rate: **~8%**

### After LoRA Fine-tuning (Target)
- Same character, different images: **0.92-0.98 similarity** ✅
- Different characters: **0.30-0.60 similarity** ✅
- False positive rate: **~2%**

### Validation Metrics
- **Character ID Accuracy**: >95% on DAF:re test set
- **Embedding Quality**: Intra-character similarity > 0.90, inter-character < 0.65
- **Stage 5 Performance**: Reduce false positives by 75%

---

## Training Requirements

### Hardware
- **Minimum**: 1x NVIDIA GPU with 16GB VRAM (RTX 4090, A4000)
- **Recommended**: 1x A100 40GB or 2x RTX 4090
- **Training Time**: 6-12 hours for 10 epochs

### Software
- Python 3.12+
- PyTorch 2.0+
- open-clip-torch
- peft (LoRA)
- transformers

### Storage
- DAF:re dataset: ~50GB
- Model checkpoints: ~2-7MB per epoch (LoRA only)
- Logs and metrics: ~100MB

### Cost Estimate
- **Cloud GPU (A100 40GB)**: $2.50/hour × 10 hours = **$25**
- **RunPod/VastAI**: $1.00-1.50/hour × 10 hours = **$10-15**

---

## Incremental Learning for New Characters

### Problem
DAF:re doesn't include recent anime (Dandadan, etc.)

### Solution: Continuous Learning Pipeline

**1. Collect New Character Data from Stage 5**
```python
# After each Stage 5 run, save character images
stage5_output = {
    "character_name": "Momo Ayase",
    "images": [url1, url2, url3, url4],  # From 4 sources
    "anime": "Dandadan"
}

# Save to incremental training set
save_for_incremental_training(stage5_output, "data/incremental/")
```

**2. Periodic Retraining**
```bash
# Every 100 new characters or monthly
uv run python scripts/retrain_character_lora.py \
    --base-weights models/character_lora/lora_weights.pt \
    --new-data data/incremental/ \
    --output models/character_lora/lora_weights_v2.pt
```

**3. Few-Shot Adaptation**
- For brand new anime with <10 characters, use few-shot learning
- Freeze base LoRA, add small adapter for new characters
- Requires only 5-10 images per character

---

## Monitoring and Evaluation

### Training Metrics
- Loss curves (train/val)
- Accuracy (top-1, top-5)
- Per-character performance
- Confusion matrix

### Stage 5 Integration Metrics
- Visual similarity distributions (same vs different characters)
- False positive rate
- True positive rate
- Processing time impact

### Logging
```python
# Weights & Biases or TensorBoard
import wandb

wandb.init(project="anime-character-recognition")
wandb.log({
    "epoch": epoch,
    "train_loss": train_loss,
    "val_acc": val_acc,
    "same_char_similarity": 0.94,
    "diff_char_similarity": 0.42
})
```

---

## Next Steps

1. **Download DAF:re dataset** and analyze structure
2. **Create training script** with LoRA integration
3. **Train baseline model** (10 epochs)
4. **Evaluate on test set** (>95% accuracy target)
5. **Integrate into Stage 5** and measure improvements
6. **Set up incremental learning** for new characters

---

## References

- DAF:re Paper: https://arxiv.org/abs/2101.08674
- LoRA: https://arxiv.org/abs/2106.09685
- OpenCLIP: https://github.com/mlfoundations/open_clip
- PEFT Library: https://github.com/huggingface/peft
