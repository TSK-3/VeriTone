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
    parser = argparse.ArgumentParser(); parser.add_argument("--backbone", required=True); parser.add_argument("--genuine-dir", required=True); parser.add_argument("--spoof-dir", required=True); parser.add_argument("--out", required=True); parser.add_argument("--epochs", type=int, default=5); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--gradient-accumulation", type=int, default=1); parser.add_argument("--lr", type=float, default=2e-5); parser.add_argument("--num-workers", type=int, default=2); parser.add_argument("--val-split", type=float, default=.15); parser.add_argument("--crops", type=int, default=1); parser.add_argument("--synthetic", type=int, default=0); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--amp", action="store_true"); parser.add_argument("--resume", help="resume from a Tier 2 checkpoint")
    args = parser.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    if args.amp and not use_amp:
        print("warning: --amp requested without CUDA; continuing in float32")
    if args.gradient_accumulation < 1:
        raise SystemExit("--gradient-accumulation must be >= 1")
    train_ds, val_ds = build_datasets(args); train = torch.utils.data.DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers); val = torch.utils.data.DataLoader(val_ds, args.batch_size, num_workers=args.num_workers)
    model = SSLBinaryDetector(args.backbone).to(device); optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01); scaler = torch.cuda.amp.GradScaler(enabled=use_amp); best, best_state, start_epoch = float("inf"), None, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint.get("backbone") != args.backbone:
            raise ValueError(f"checkpoint backbone {checkpoint.get('backbone')!r} does not match {args.backbone!r}")
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best = float(checkpoint.get("best_val_loss", best))
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(f"resuming {args.resume} at epoch {start_epoch + 1}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        for step, (wave, label) in enumerate(train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = nn.functional.binary_cross_entropy_with_logits(model(wave.to(device)), label.to(device)) / args.gradient_accumulation
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation == 0 or step + 1 == len(train):
                scaler.unscale_(optimiser); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); scaler.step(optimiser); scaler.update(); optimiser.zero_grad(set_to_none=True)
        model.eval(); total = count = 0
        with torch.no_grad():
            for wave, label in val:
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = nn.functional.binary_cross_entropy_with_logits(model(wave.to(device)), label.to(device))
                total += loss.item()*len(label); count += len(label)
        score = total/max(count,1); print(f"epoch={epoch+1} val_loss={score:.4f}")
        state = {"backbone":args.backbone, "model_state_dict":model.state_dict(), "optimizer_state_dict":optimiser.state_dict(), "epoch":epoch, "best_val_loss":best}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True); torch.save(state, f"{args.out}.last")
        if score < best: best, best_state = score, {k:v.detach().cpu() for k,v in model.state_dict().items()}; torch.save({"backbone":args.backbone,"model_state_dict":best_state,"epoch":epoch,"best_val_loss":best}, args.out)
    if best_state is None:
        raise RuntimeError("no checkpoint was produced")
    print(f"saved {args.out}; latest checkpoint {args.out}.last")
    return 0
if __name__ == "__main__": raise SystemExit(main())
