# Edge Acoustic Monitoring

# Environmental Sound Classification

## Overview
This project focuses on environmental sound classification using raw audio data.  
The current target classes are:

- Aircraft
- Fire
- Thunder
- Train
- Water
- Wind

We use lightweight audio classification models for comparison:
- BC-ResNet
- MatchboxNet

## Data Pipeline
The current training pipeline uses raw audio files referenced by a metadata CSV file.

- Metadata file: `data/metadata/master_audio_dataset_no_negative.csv`
- Audio root: `data/segments/`
- Input audio is standardized to:
  - 16 kHz
  - mono
  - 5 seconds
- Features are generated on the fly as 64-bin log-mel spectrograms

## Training Scripts
Recommended training scripts:

- `train_bcresnet_results_bundle.py`
- `train_matchboxnet_results_bundle.py`

Older scripts such as `train_bcresnet.py` and `train_matchboxnet.py` were based on earlier preprocessing pipelines and are kept only for reference if needed.

## Results Output
Each training run automatically creates a timestamped results folder containing:

- best model checkpoint
- training log
- classification report
- confusion matrix
- test predictions
- config file
- split metadata used

## Run Training
```bash
python train_bcresnet_results_bundle.py
python train_matchboxnet_results_bundle.py

## Legacy Note
An earlier baseline version of this project used a 4-class setup with:
- Fire
- Thunder
- Water
- Negative

using 2-second ESC-50-based segments and an earlier BC-ResNet baseline pipeline.  
Those older scripts are kept only for reference and are no longer the main training workflow.