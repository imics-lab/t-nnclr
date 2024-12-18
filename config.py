import torch

# Model architecture
INPUT_SHAPE = (96, 4)  # (sequence_length, channels)
EMBEDDING_DIM = 64     # Size of the output embedding vector
N_CLASSES = 6         # Number of classes in the dataset

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
TEMPERATURE = 0.1        # Temperature for contrastive loss
QUEUE_SIZE = 1000       # Size of the nearest neighbor queue

# Training epochs
PRETRAIN_EPOCHS = 2   
FINETUNE_EPOCHS = 2

# Encoder architecture params
CONV_FILTERS = 100
KERNEL_SIZE = 16
DROPOUT_RATE = 0.5
POOL_SIZE = 2

# Data processing
SHUFFLE_BUFFER_SIZE = 1000

# Class names
CLASS_NAMES = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Augmentation parameters
JITTER_SIGMA = 0.03
SCALING_SIGMA = 0.1
TIMEWARPING_SIGMA = 0.2
TIMEWARPING_KNOT = 4