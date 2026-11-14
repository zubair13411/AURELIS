import argparse
from PIL import Image, UnidentifiedImageError
from src.inference import load_model, predict_image


def main():
    parser = argparse.ArgumentParser(description='Run a research-only AURELIS prediction.')
    parser.add_argument('image', help='Path to a chest X-ray image file')
    parser.add_argument('--model', default='models/best_model.pt')
    args = parser.parse_args()
    model = load_model(args.model)
    try:
        with Image.open(args.image) as image:
            result = predict_image(model, image.convert('RGB'))
    except (OSError, UnidentifiedImageError) as exc:
        parser.error(f'Could not read image: {exc}')
    print(result)


if __name__ == '__main__':
    main()
