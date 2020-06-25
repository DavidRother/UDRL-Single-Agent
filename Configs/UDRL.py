# Number of iterations in the main loop
MAIN_LOOP_ITERATIONS = 5000

# Number of (input, target) pairs per batch used for training the behavior function
BATCH_SIZE = 1024

# Scaling factor for desired horizon input
HORIZON_SCALE = 0.01

# Scaling factor for desired return input
RETURN_SCALE = 0.01

# Number of episodes from the end of the replay buffer used for sampling exploratory
# commands
LAST_FEW = 50

# Learning rate for the ADAM optimizer
LEARNING_RATE = 1e-3

# Number of exploratory episodes generated per step of UDRL training
NUM_EXPLORATORY_EPISODES = 30

# Number of gradient-based updates of the behavior function per step of UDRL training
NUM_UPDATES_PER_TRAINING_EPOCH = 100

# Number of warm up episodes at the beginning of training
NUM_WARM_UP_EPISODES = 20

# Maximum size of the replay buffer (in episodes)
REPLAY_BUFFER_SIZE = 500

# Evaluate the agent after `evaluate_every` iterations
EVALUATION_FREQUENCY = 20

# Maximum steps allowed
MAX_STEPS = 1000

# Times we evaluate the agent
NUM_EVALUATIONS = 100

# Epsilon for epsilon greedy action selection
EPSILON_GREEDY = 0.05
