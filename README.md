# Edge Acoustic Monitoring

Phase 1 baseline for edge-friendly acoustic event monitoring.

## Scope (Phase 1)
4-class short-window audio classification (2s window):
- Fire
- Thunder
- Water
- Negative (wind / train / airplane / helicopter)

Model:
- Lightweight BC-ResNet-style CNN on log-mel spectrogram features

## Repository layout
- `data/metadata/`  
  Small CSV metadata used to reproduce splits and training.
- `data/results/`  
  Saved evaluation outputs (e.g., confusion matrix, metrics).
- `data/raw/`, `data/segments/`, `data/features/`  
  Not tracked in Git (large files). You must download datasets locally.

## Datasets
Phase 1 uses ESC-50:
- https://github.com/karolpiczak/ESC-50

Expected local path:
- `data/raw/esc50/ESC-50-master/`

## Environment
Python 3.10 recommended.

Install dependencies:
```bash
python -m pip install -U pip
python -m pip install pandas numpy librosa soundfile tqdm matplotlib
python -m pip install torch torchvision torchaudio