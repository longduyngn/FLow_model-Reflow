import os

# Data parameters
DATASET_TYPE = "checkerboard"  # Options: "gaussian_8mode", "two_moons", "swiss_roll", "checkerboard"
NUM_SAMPLES_DATA = 100000
NUM_SAMPLES_REFLOW = 10000
RADIUS = 5.0
STD = 0.5

# Model parameters
HIDDEN_DIM = 256
TIME_EMBED_DIM = 64

# Training parameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS_FM = 120
EPOCHS_REFLOW = 120
PRINT_INTERVAL = 20

# Reflow pair creation mode: "forward" or "backward"
REFLOW_PAIR_MODE = "backward"

# Evaluation parameters
ODE_STEPS = 20

# Directories
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"
DATA_DIR = "data"

# Make sure directories exist (if run locally)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
