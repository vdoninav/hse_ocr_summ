# Учитель
TEACHER_MODEL_NAME = "models/sota"

# Student
EMBED_DIM = 512
ENC_HIDDEN_DIM = 256
DEC_HIDDEN_DIM = 512
DROPOUT = 0.3
NUM_LAYERS = 2

# Общие гиперпараметры
MAX_SOURCE_LEN = 1024
MAX_TARGET_LEN = 128

# Обучение
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
TEMPERATURE = 2.0

# Beam Search
BEAM_SIZE = 8
BEAM_MAX_LENGTH = 128
MIN_LEN = 10

# Wandb
WANDB_PROJECT = "coursework_2025"
WANDB_RUN_NAME = "v3_2"
