import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import AASIST
from dataset import ASVSpoofDataset

def train_model(train_dir, protocol, epochs=10, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_set = ASVSpoofDataset(train_dir, protocol)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model, loss, optimizer
    model = AASIST().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for lfcc, labels in loader:
            lfcc, labels = lfcc.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(lfcc)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * lfcc.size(0)

        avg_loss = running_loss / len(train_set)
        print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")
        # Save a checkpoint each epoch
        torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pt")

if __name__ == "__main__":
    import sys
    train_dir = sys.argv[1]       # e.g. "data/train_wav"
    protocol = sys.argv[2]        # e.g. "protocol/ASVspoof2019_LA_train.trl"
    train_model(train_dir, protocol)
