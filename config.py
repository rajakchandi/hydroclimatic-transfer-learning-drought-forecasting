from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "data" / "raw" / "source_basin.csv"
TARGET_CSV = ROOT / "data" / "raw" / "target_basin.csv"
DATE_COLUMN = "Date"
TARGET_COLUMN = "SPEI_6"

FEATURE_COLUMNS = [
    "Precip", "Tmax", "Tmin", "PET", "Vapour_pressure",
    "Wind_speed", "Soil_moisture", "SRAD",
    "SOI", "El_nino3.4", "PDO", "IOD",
]

LEAD_TIMES = [1, 3, 6]
SEQUENCE_LENGTH = 12
TARGET_ADAPTATION_FRACTIONS = [0.20, 0.30]
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.15

BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 32
CNN_CHANNELS = 32
DROPOUT = 0.25
L2_SP_LAMBDA = 1e-3
RANDOM_SEED = 42
DEVICE = "cuda"  # automatically falls back to CPU when CUDA is unavailable

OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
