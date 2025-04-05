# preprocess.py

import os
import torchaudio

def preprocess_audio(input_path, output_path, target_sr=16000):
    """
    Convert an audio file to 16 kHz mono WAV.
    """
    waveform, sr = torchaudio.load(input_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    resampler = torchaudio.transforms.Resample(sr, target_sr)
    resampled = resampler(waveform)
    torchaudio.save(output_path, resampled, target_sr)

def batch_preprocess(src_dir, dst_dir, ext_in=".flac", ext_out=".wav"):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if not fname.endswith(ext_in):
            continue
        in_path = os.path.join(src_dir, fname)
        out_fname = os.path.splitext(fname)[0] + ext_out
        out_path = os.path.join(dst_dir, out_fname)
        preprocess_audio(in_path, out_path)

if __name__ == "__main__":
    # Example usage:
    # python preprocess.py /path/to/flac /path/to/wav
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    batch_preprocess(src, dst)
