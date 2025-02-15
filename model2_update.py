import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout
from collections import deque, defaultdict
import random

# User Input
user_input = input("Enter the opcode sequence as comma-separated values (e.g., 97,0,216,97,2,174,86,91): ")
input_sequence = list(map(int, user_input.split(',')))

# Training Data (Opcode Sequences with Labels)
training_data = [
    (np.array([96, 128, 96, 64, 82, 52, 128, 21, 96, 14, 87]), 1),  # Reentrancy
    (np.array([95, 53, 96, 224, 28, 128, 99, 18, 6, 95]), 0),  # Overflow
    (np.array([99, 208, 227, 13, 176, 20, 97, 0, 208, 87]), 3),  # Unauthorized Access
    (np.array([115, 255, 255, 255, 255, 255, 255, 255]), 4),  # Gas Efficiency
    (np.array([97, 0, 216, 97, 2, 174, 86, 91]), 5),  # Self-Destruct
    (np.array([95, 96, 32, 82, 128, 95, 82, 96]), 2)  # Frontrunning
]


for seq, label in training_data:

    existing_sequences = [td[0] for td in training_data]

if not any(np.array_equal(input_sequence, seq) for seq in existing_sequences):
    print("No vulnerabilty detected")

    # Log it and add it to dataset
    #training_data.append((input_sequence, 6))  # Use -1 for unknown class
else:
    print("Vulnerability detected.")
    # Model Parameters
    state_size = max(len(seq) for seq, _ in training_data)
    action_size = 6  # Number of vulnerability classes
    vocab_size = 300
    embedding_dim = 128

    def pad_sequence(seq, max_len):
        """Pad sequences to match the longest sequence length."""
        return np.pad(seq, (0, max_len - len(seq)), 'constant')

    # Preprocess Training Data
    training_data = [(pad_sequence(seq, state_size), label) for seq, label in training_data]

    class DQNAgent:
        def __init__(self, state_size, action_size, vocab_size, embedding_dim):
            self.state_size = state_size
            self.action_size = action_size
            self.memory = deque(maxlen=2000)
            self.gamma = 0.95
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.learning_rate = 0.001
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.model = self._build_model()

        def _build_model(self):
            """Build the neural network model."""
            model = Sequential()
            model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.state_size))
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(self.action_size, activation='softmax'))  # Changed to softmax for classification
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model

        def remember(self, state, action, reward, next_state, done):
            """Store experiences for replay."""
            self.memory.append((state, action, reward, next_state, done))

        def act(self, state):
            """Select an action based on the current state."""
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])

        def replay(self, batch_size):
            """Train the model using experience replay."""
            if len(self.memory) < batch_size:
                return
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = reward + self.gamma * np.max(self.model.predict(next_state, verbose=0)[0])
                target_f = self.model.predict(state, verbose=0)
                target_f[0][action] = target
                self.model.fit(state, np.argmax(target_f, axis=1), epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        def load_weights(self, name):
            self.model.load_weights(name)

        def save_weights(self, name):
            self.model.save_weights(name)

    # Initialize Agent
    agent = DQNAgent(state_size, action_size, vocab_size, embedding_dim)

    def process_input(opcode_sequence):
        """Classify the input opcode sequence."""
        opcode_sequence = pad_sequence(np.array(opcode_sequence), state_size).reshape(1, state_size)
        action = agent.act(opcode_sequence)
        return action

    # Training
    epochs = 40
    batch_size = 32

    for e in range(epochs):
        for opcode_sequence, correct_label in training_data:
            state = opcode_sequence.reshape(1, state_size)
            action = agent.act(state)
            reward = 1 if action == correct_label else -1
            next_state = state
            done = True
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        print(f"Episode {e+1}/{epochs} - Epsilon: {agent.epsilon:.4f}")

    # Test Classification
    predicted_class = process_input(input_sequence)
    print(f"Predicted class for the input: {predicted_class}")

    # Save Model
    agent.save_weights('dqn_modnn_vulnerability_detection.weights.h5')


