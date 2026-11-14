from pathlib import Path
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-4
DATA_DIR = Path('data/chest_xray')
MODEL_DIR = Path('models')
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
SEED = 42
