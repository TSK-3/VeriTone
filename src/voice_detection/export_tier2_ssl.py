"""Export a fine-tuned SSL Tier 2 checkpoint to the strict ONNX serving contract."""
from __future__ import annotations
import argparse
import torch
from .train_tier2_ssl import SSLBinaryDetector

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--out", required=True); parser.add_argument("--opset", type=int, default=17); args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True); model = SSLBinaryDetector(checkpoint["backbone"]); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    sample = torch.zeros(1, 24_000)
    torch.onnx.export(model, sample, args.out, input_names=["waveform"], output_names=["spoof_logit"], dynamic_axes={"waveform": {1:"samples"}}, opset_version=args.opset)
    print(f"exported {args.out}; validate with config/tier2-manifest.example.json")
    return 0
if __name__ == "__main__": raise SystemExit(main())
