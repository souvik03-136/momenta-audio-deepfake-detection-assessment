# Audio Deepfake Detection with AASIST

This repository implements a simplified AASIST-based pipeline for audio deepfake detection on the ASVspoof 2019 Logical Access (LA) dataset. It includes scripts to:

1. **Preprocess** raw `.flac` files into 16 kHz mono WAVs
2. **Extract** LFCC features and load data via a PyTorch `Dataset`
3. **Train** the AASIST model
4. **Evaluate** the model and compute EER on held-out data

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites & Dependencies](#prerequisites--dependencies)
- [Data Acquisition](#data-acquisition)
- [Setup & Installation](#setup--installation)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference on New Audio](#inference-on-new-audio)
- [Reproducibility Notes](#reproducibility-notes)

---

## Project Structure

```
asvspoof_project/
├── data/
│   ├── raw_flac/           # Original ASVspoof2019 LA .flac files
│   ├── train_wav/          # 16 kHz mono WAVs for training
│   ├── dev_wav/            # 16 kHz mono WAVs for development
│   └── eval_wav/           # 16 kHz mono WAVs for evaluation
│
├── protocol/
│   ├── ASVspoof2019_LA_train.trl
│   ├── ASVspoof2019_LA_dev.trl
│   └── ASVspoof2019_LA_eval.trl
│
├── models/                 # Saved model checkpoints
│   └── checkpoint_epoch10.pt
│
├── preprocess.py           # Batch audio conversion script
├── dataset.py              # PyTorch Dataset for LFCC features
├── model.py                # AASIST model definition
├── train.py                # Training script
├── eval.py                 # Evaluation & EER calculation
│
├── requirements.txt        # Python dependencies
└── README.md               # This documentation
```

---

## Prerequisites & Dependencies

- **Python** 3.8 or later
- **PyTorch** ≥1.10.0 with CUDA (optional, for GPU acceleration)
- **torchaudio** ≥0.10.0
- **pandas**, **numpy**, **scikit-learn**

Install all Python packages with:

```bash
pip install -r requirements.txt
```

---

## Data Acquisition

1. Visit the official ASVspoof 2019 LA challenge page: https://www.asvspoof.org/
2. Register (if required) and download the Logical Access (LA) partitions:
   - Training set (FLAC)
   - Development set (FLAC)
   - Evaluation set (FLAC)
3. Place the extracted `.flac` files under:

   ```
   data/raw_flac/
   ```
4. Ensure you also have the protocol files (provided by ASVspoof) in `protocol/`:
   - `ASVspoof2019_LA_train.trl`
   - `ASVspoof2019_LA_dev.trl`
   - `ASVspoof2019_LA_eval.trl`

---

## Setup & Installation

1. **Clone this repository**
   ``` bash
   git clone https://github.com/souvik03-136/momenta-audio-deepfake-detection-assessment.git
   
   cd momenta-audio-deepfake-detection-assessment.git
   ```

2. **Create and activate a virtual environment** (optional but recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Preprocessing

Convert raw FLAC files to 16 kHz mono WAVs for each split.

```bash
python preprocess.py data/raw_flac/ data/train_wav/
python preprocess.py data/raw_flac/ data/dev_wav/
python preprocess.py data/raw_flac/ data/eval_wav/
```

This script:
- Loads each input file with `torchaudio`
- Resamples to 16 kHz
- Converts to mono if necessary
- Saves as `.wav` in the target directory

---

## Training

Train the AASIST model on the training split.

```bash
python train.py data/train_wav/ protocol/ASVspoof2019_LA_train.trl
```

- **Epochs**: 10 (default)
- **Batch size**: 32
- **Learning rate**: 1e-3

Checkpoints are saved as `checkpoint_epoch{n}.pt` in the working directory.

---

## Evaluation

Compute Equal Error Rate (EER) on the evaluation split:

```bash
python eval.py models/checkpoint_epoch10.pt data/eval_wav/ protocol/ASVspoof2019_LA_eval.trl
```

Output:
```
EER: X.XX%
```

---

## Inference on New Audio

Use the trained model to classify a single WAV file:

```python
from model import AASIST
import torchaudio, torch

# Load model
model = AASIST()
model.load_state_dict(torch.load("models/checkpoint_epoch10.pt"))
model.eval()

# Load and preprocess
waveform, sr = torchaudio.load("path/to/file.wav")
lfcc = torchaudio.compliance.kaldi.lfcc(waveform, num_ceps=60, sample_frequency=sr)
lfcc = lfcc.transpose(0,1).unsqueeze(0)

# Predict
with torch.no_grad():
    logits = model(lfcc)
    pred = logits.argmax(dim=1).item()
    print("bonafide" if pred==0 else "spoof")
```

---

## Reproducibility Notes

- **Data splits**: Must match ASVspoof 2019 LA partitions exactly. Do not shuffle speakers between train/dev/eval.
- **Random seeds**: For exact replication, set `torch.manual_seed(...)` at the top of `train.py` and `numpy.random.seed(...)`.
- **Hardware**: Training on GPU (e.g., NVIDIA RTX series) will complete 10 epochs in ~30 minutes. CPU training is significantly slower.

---

For questions or issues, please open a GitHub issue or contact the maintainer at `souvikmahanta2003@gmail.com`.

