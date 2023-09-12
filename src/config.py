DATA_DIR = "ball_detection/"
NUM_CLASSES = 2
CLASS_WEIGHTING= [ 0.5, 0.5]
RESIZE_FACTOR = 4 # Greates common devisor
NUM_CHANNELS = 4
IMAGE_HEIGHT = int(768 / RESIZE_FACTOR)
IMAGE_WIDTH = int(1360 / RESIZE_FACTOR)
DATA_MEAN = (0.1075, 0.1197, 0.0787)
DATA_STD = (0.0441, 0.0449, 0.0513)

# Training hyperparameters
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS
LEARNING_RATE = 0.002
BATCH_SIZE = 4
NUM_WORKERS = 4
MIN_EPOCHS = 1
MAX_EPOCHS = 3

# Compute related
ACCELERATOR = "cpu"
PRECISION = 'bf16-mixed'
LOGS_FOLDER = 'logs'


