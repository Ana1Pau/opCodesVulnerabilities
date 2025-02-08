import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout
from collections import deque, defaultdict
import random
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Example opcode sequences with associated vulnerabilities
training_data = [
    (np.array([96, 128, 96, 64, 82, 52, 128, 21, 96, 14, 87]), 1),  # Reentrancy
    (np.array([95, 53, 96, 224, 28, 128, 99, 18, 6, 95]), 0),  # Overflow
    (np.array([99, 208, 227, 13, 176, 20, 97, 0, 208, 87]), 3),  # Unauthorized Access
    (np.array([115, 255, 255, 255, 255, 255, 255, 255]), 4),  # Gas Efficiency
    (np.array([97, 0, 216, 97, 2, 174, 86, 91]), 5),  # Self-Destruct
    (np.array([95, 96, 32, 82, 128, 95, 82, 96]), 2)  # Frontrunning
]

# Model Parameters
state_size = max(len(seq) for seq, _ in training_data)  # Max opcode sequence length
global action_size
action_size = 6  # Number of vulnerability classes
vocab_size = 300
embedding_dim = 128
input_length = state_size

# Dictionary to track unknown vulnerabilities
unknown_vulnerabilities = defaultdict(int)
new_vulnerabilities = {}

def pad_sequence(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)), 'constant')

training_data = [(pad_sequence(seq, state_size), label) for seq, label in training_data]

class DQNAgent:
    def __init__(self, state_size, action_size, vocab_size, embedding_dim, input_length):
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
        self.input_length = input_length
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def load(self, name):
        self.model.build((None, self.input_length))  # Ensure model is built
        self.model.load_weights(name)

agent = DQNAgent(state_size, action_size, vocab_size, embedding_dim, input_length)
agent.load('dqn_modnn_vulnerability_detection.weights.h5')
agent.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=agent.learning_rate), loss='mse')


def process_input(opcode_sequence):
    opcode_sequence = pad_sequence(np.array(opcode_sequence), state_size).reshape(1, state_size)
    action = agent.act(opcode_sequence)
    return action

ep = 150  # Training episodes
batch_size = 32

# Training loop
for e in range(ep):
    for opcode_sequence, correct_label in training_data:
        state = opcode_sequence.reshape(1, state_size)
        action = agent.act(state)
        
        reward = 1 if action == correct_label else -1
        next_state = state  # Simplified state transition
        done = True  # Episode ends after one step
        agent.remember(state, action, reward, next_state, done)
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    
    #print(f"Episode {e}/{ep} - Predicted Class: {action} - Correct Class: {correct_label} - Epsilon: {agent.epsilon}")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'opcode_sequence' not in data:
        return jsonify({'error': 'Missing opcode_sequence'}), 400
    opcode_sequence = data['opcode_sequence']
    prediction = process_input(opcode_sequence)
    return jsonify({'predicted_class': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True, port=8170)
