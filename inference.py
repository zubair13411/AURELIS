
import os
import torch
from pathlib import Path
from urllib.request import urlretrieve
from torchvision import transforms

from model import build_model
from config import IMAGE_SIZE, CLASS_NAMES

MODEL_URL = (
    "https://huggingface.co/zubair13411/AURELIS/"
    "resolve/main/aurelis_resnet18.pth"
)

MODEL_CACHE = Path("models/best_model.pt")


def download_model():
    MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_CACHE.exists():
        print("Downloading AURELIS model...")
        urlretrieve(MODEL_URL, MODEL_CACHE)
        print("Model download complete.")

    return MODEL_CACHE


def load_model(path=None, device=None):
    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_path = Path(path) if path else download_model()

    model = build_model(pretrained=False)

    state = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state)
    model.to(device).eval()

    return model


def predict_image(model, image):
    device = next(model.parameters()).device

    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    x = tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    idx = int(probs.argmax().item())
    confidence = float(probs[idx].item())

    if confidence < 0.65:
        return {
            "label": "UNCERTAIN — HUMAN REVIEW NEEDED",
            "confidence": confidence,
            "message": (
                "The model confidence is below "
                "the prototype review threshold."
            )
        }

    return {
        "label": CLASS_NAMES[idx],
        "confidence": confidence,
        "message": (
            "Experimental research output only; "
            "not a clinical conclusion or diagnosis."
        )
    }
