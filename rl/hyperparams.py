# Hyperparameters
NUM_ITERATIONS = 1
EPISODES = 1 # Number of episodes to train on
# STEPS_PER_EPISODE = 1000 # Number of steps per each episode
EPSILON_START = 1.0  # Starting value of epsilon:
EPSILON_END = 0.10  # Minimum value of epsilon
EPSILON_DECAY = 0.999  # Decay rate of epsilon per episode (possibly too slow): 0.999
EPSILON=0.15 # only relevant if decay is not used
GAMMA = 0.10  # Discount factor for future rewards (inspect)
LEARNING_RATE = 0.001  # Learning rate
LEARNING_RATE_DECAY = 0.99  # Decay learning rate each episode to stabilize training later
BATCH_SIZE = 8 # Batch size for experience replay sampling 
TARGET_UPDATE = 16 # Update target network parameters every 16 steps
PERFORMANCE_FACTOR = 0.8 # Reward scheme