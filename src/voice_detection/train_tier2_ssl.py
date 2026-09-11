"""Fine-tune XLS-R or WavLM anti-spoof heads on labelled 16 kHz WAV folders.

Example (run on a Colab T4/A10/L4 GPU)::

  python -m voice_detection.train_tier2_ssl --backbone facebook/wav2vec2-xls-r-300m \
    --genuine-dir data/genuine --spoof-dir data/spoof --out checkpoints/wav2vec2_xlsr.pt

Use the same command with ``microsoft/wavlm-large`` for the WavLM member. The
dataset split must hold out complete spoofing-generator families, not random
clips, to measure unseen-attack generalisation honestly.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModel
from .train_tier1 import build_datasets

class SSLBinaryDetector(nn.Module):
    def __init__(self, backbone: str, dropout: float = .2) -> None:
        super().__init__(); self.backbone_name = backbone; self.encoder = AutoModel.from_pretrained(backbone)
        self.head = nn.Sequential(nn.LayerNorm(self.encoder.config.hidden_size), nn.Dropout(dropout), nn.Linear(self.encoder.config.hidden_size, 1))
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_values=wave).last_hidden_state
        return self.head(hidden.mean(dim=1)).squeeze(-1)

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--backbone", required=True); parser.add_argument("--genuine-dir", required=True); parser.add_argument("--spoof-dir", required=True); parser.add_argument("--out", required=True); parser.add_argument("--epochs", type=int, default=5); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--lr", type=float, default=2e-5); parser.add_argument("--num-workers", type=int, default=2); parser.add_argument("--val-split", type=float, default=.15); parser.add_argument("--crops", type=int, default=1); parser.add_argument("--synthetic", type=int, default=0); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = build_datasets(args); train = torch.utils.data.DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers); val = torch.utils.data.DataLoader(val_ds, args.batch_size, num_workers=args.num_workers)
    model = SSLBinaryDetector(args.backbone).to(device); optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01); best, best_state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        for wave, label in train:
            optimiser.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(model(wave.to(device)), label.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimiser.step()
        model.eval(); total = count = 0
        with torch.no_grad():
            for wave, label in val:
                loss = nn.functional.binary_cross_entropy_with_logits(model(wave.to(device)), label.to(device)); total += loss.item()*len(label); count += len(label)
        score = total/max(count,1); print(f"epoch={epoch+1} val_loss={score:.4f}")
        if score < best: best, best_state = score, {k:v.cpu() for k,v in model.state_dict().items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); torch.save({"backbone":args.backbone,"model_state_dict":best_state}, args.out); print(f"saved {args.out}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
