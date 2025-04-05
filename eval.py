import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
import numpy as np
from models import AASIST
from dataset import ASVSpoofDataset

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=0)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return fpr[idx]

def evaluate(model_path, data_dir, protocol, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load data
    eval_set = ASVSpoofDataset(data_dir, protocol)
    loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False, num_workers=4)
    # Load model
    model = AASIST().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels, all_scores = [], []
    with torch.no_grad():
        for lfcc, labels in loader:
            lfcc = lfcc.to(device)
            logits = model(lfcc)
            # use the logit for the 'bonafide' class (index 0) as score
            scores = logits[:,0].cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())

    eer = calculate_eer(np.array(all_labels), np.array(all_scores))
    print(f"EER: {eer*100:.2f}%")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]    # e.g. "checkpoint_epoch10.pt"
    data_dir   = sys.argv[2]    # e.g. "data/eval_wav"
    protocol   = sys.argv[3]    # e.g. "protocol/ASVspoof2019_LA_eval.trl"
    evaluate(model_path, data_dir, protocol)
