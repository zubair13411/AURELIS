import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from src.model import build_model
from src.data import make_loaders
from src.config import EPOCHS, LEARNING_RATE, MODEL_DIR, SEED


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(epochs=EPOCHS):
    if epochs < 1:
        raise ValueError('epochs must be at least 1')
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dl, val_dl, _, classes = make_loaders()
    model = build_model(pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_val = float('inf')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(device), labels.to(device)
                val_loss += criterion(model(images), labels).item() * labels.size(0)
        train_loss /= len(train_dl.dataset)
        val_loss /= len(val_dl.dataset)
        print(f'Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_DIR / 'best_model.pt')
            metadata = {'architecture': 'ResNet-18', 'classes': classes,
                        'image_size': 224, 'best_validation_loss': best_val,
                        'epochs_requested': epochs, 'seed': SEED}
            (MODEL_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    print(f'Best checkpoint saved to {MODEL_DIR / "best_model.pt"}')
    print(f'Classes: {classes}; device: {device}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AURELIS ResNet-18 research model.')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    run(parser.parse_args().epochs)
