# dataset.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio

class ASVSpoofDataset(Dataset):
    """
    ASVspoof 2019 LA dataset loader.
    Protocol files list: <utt_id> <system_id> <key> <...>
    We assume protocol format: utt_id, _, _, key
    """
    def __init__(self, root_dir, protocol_file, ext=".wav"):
        self.root_dir = root_dir
        cols = ["utt_id", "system_id", "key"]
        self.protocol = pd.read_csv(protocol_file, sep=" ", names=cols, usecols=[0,1,2])
        self.ext = ext

    def __len__(self):
        return len(self.protocol)

    def __getitem__(self, idx):
        row = self.protocol.iloc[idx]
        utt = row.utt_id
        label = 0 if row.key == "bonafide" else 1
        audio_path = os.path.join(self.root_dir, utt + self.ext)
        waveform, sr = torchaudio.load(audio_path)
        # Compute LFCC features (60 dims)
        lfcc = torchaudio.compliance.kaldi.lfcc(
            waveform, num_ceps=60, sample_frequency=sr
        )
        # lfcc: (time, 60) → transpose to (1, 60, time)
        lfcc = lfcc.transpose(0,1).unsqueeze(0)
        return lfcc, torch.tensor(label, dtype=torch.long)
