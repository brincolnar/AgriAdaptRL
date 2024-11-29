# Hyperparameters
NUM_ITERATIONS = 1
EPISODES = 1 # Number of episodes to train on
# STEPS_PER_EPISODE = 1000 # Number of steps per each episode
EPSILON_START = 1.0  # Starting value of epsilon:
EPSILON_END = 0.01  # Minimum value of epsilon
EPSILON_DECAY = 0.99  # standard for 497 is 0.99 Try faster decay 0.98, slower decay 0.995
EPSILON=0.10 # only relevant if decay is not used
GAMMA = 0.95  # Discount factor for future rewards
LEARNING_RATE = 0.001  # Learning rate
LEARNING_RATE_DECAY = 0.99  # Decay learning rate each episode to stabilize training later
BATCH_SIZE = 64 # standard 64 Batch size for experience replay sampling 
TARGET_UPDATE = 50 # standard 50 Update target network parameters 
INFERENCE_DIM = (512, 512)
