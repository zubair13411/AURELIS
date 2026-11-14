import json
from pathlib import Path
import torch
from src.inference import load_model
from src.data import make_loaders
from src.metrics import calculate_metrics


def main():
    _, _, test_dl, classes = make_loaders()
    if classes != ['NORMAL', 'PNEUMONIA']:
        raise ValueError(f'Unexpected class order: {classes}')
    model = load_model('models/best_model.pt')
    device = next(model.parameters()).device
    truth, preds = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_dl:
            logits = model(images.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
            truth.extend(labels.tolist())
    results = calculate_metrics(truth, preds)
    baseline = calculate_metrics(truth, [1] * len(truth))
    output = {
        'model': {k: v for k, v in results.items() if k != 'classification_report'},
        'always_predict_pneumonia_baseline': {
            k: v for k, v in baseline.items() if k != 'classification_report'
        },
        'classification_report': results['classification_report'],
        'test_images': len(truth),
        'class_order': classes,
        'warning': 'Research benchmark only; not clinical validation.'
    }
    print(json.dumps(output, indent=2))
    Path('outputs').mkdir(exist_ok=True)
    Path('outputs/test_metrics.json').write_text(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
