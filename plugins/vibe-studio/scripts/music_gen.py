#!/usr/bin/env python3
"""Generate an original, royalty-free instrumental track with ACE-Step (Apache-2.0), locally.

  music_gen.py --prompt "luxury deep house, warm piano, 112 bpm, elegant, instrumental" --seconds 30 --out track.wav [--seed 7]

Runs in its own environment (~/.local/share/vibekit/acestep-venv) because ACE-Step needs different library versions.
Prompt tips: genre + 2-3 instruments + bpm + mood. Always instrumental under voiceovers.
"""
import argparse, os, sys, shutil, glob
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", required=True)
ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--steps", type=int, default=27)
a = ap.parse_args()

import torch
import torchaudio
import soundfile as sf


def _save(path, tensor, sample_rate=48000, *args, **kw):
    """Save with soundfile (newer torchaudio needs the separate torchcodec package)."""
    arr = tensor.detach().float().cpu().numpy()
    sf.write(str(path), arr.T if arr.ndim == 2 else arr, int(sample_rate))


torchaudio.save = _save
from acestep.pipeline_ace_step import ACEStepPipeline

ckpt = Path(os.environ.get("VIBEKIT_HOME", Path.home() / ".local/share/vibekit")) / "models" / "ace-step"
ckpt.mkdir(parents=True, exist_ok=True)
dtype = "float32" if torch.backends.mps.is_available() else "bfloat16"
pipe = ACEStepPipeline(checkpoint_dir=str(ckpt), dtype=dtype, torch_compile=False)
out = Path(a.out).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
res = pipe(format="wav", audio_duration=a.seconds, prompt=a.prompt, lyrics="[instrumental]",
           infer_step=a.steps, guidance_scale=15.0, manual_seeds=[a.seed], save_path=str(out))
# the pipeline returns/saves paths; normalise to the requested filename
paths = [p for p in (res if isinstance(res, (list, tuple)) else [res]) if isinstance(p, str) and p.endswith(".wav")]
if paths and Path(paths[0]).resolve() != out:
    shutil.move(paths[0], out)
print(str(out))
