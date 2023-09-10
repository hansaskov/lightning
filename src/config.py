# Dataset
DATA_DIR = "kick_detection_case/"
TRAIN_VAL_TEST_SPLIT = [0.7, 0.2, 0.1]
RESIZE_FACTOR = 16 # Greates common devisor
IMAGE_WIDTH = int(1360 / RESIZE_FACTOR)
IMAGE_HEIGHT = int(768 / RESIZE_FACTOR)

# Training hyperparameters
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3
NUM_CLASSES = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_WORKERS = 4
MIN_EPOCHS = 1
MAX_EPOCHS = 3

# Compute related
ACCELERATOR = "cpu"
PRECISION = 'bf16-mixed'
LOGS_FOLDER = 'logs'


