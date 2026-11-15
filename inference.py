from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_URL = (
    "https://huggingface.co/zubair13411/AURELIS/"
    "resolve/main/aurelis_resnet18.onnx"
)

MODEL_PATH = Path("/tmp/aurelis_resnet18.onnx")

IMAGE_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def _ensure_model():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1_000_000:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(MODEL_URL, MODEL_PATH)

    return str(MODEL_PATH)


def load_session():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )

    return ort.InferenceSession(
        _ensure_model(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def predict_image(session, image):
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

    arr = np.asarray(image, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None, ...]

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: arr})[0][0].astype(np.float64)

    logits -= np.max(logits)
    probs = np.exp(logits) / np.sum(np.exp(logits))

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    label = (
        "UNCERTAIN — HUMAN REVIEW NEEDED"
        if confidence < 0.65
        else CLASS_NAMES[idx]
    )

    return {
        "label": label,
        "confidence": confidence,
        "message": (
            "Experimental model output only. It is not a diagnosis, "
            "clinical conclusion, treatment recommendation, or triage decision."
        ),
    }
