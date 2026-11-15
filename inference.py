
import torch
from torchvision import transforms

from model import build_model
from config import IMAGE_SIZE, CLASS_NAMES


def load_model(path, device=None):
    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_model(pretrained=False)

    state = torch.load(
        path,
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
            "Research prototype output only; "
            "not a clinical conclusion."
        )
    }
