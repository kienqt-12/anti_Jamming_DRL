from environment import environment
import numpy as np
from parameters import *
import tensorflow as tf
from keras.layers import Dense, Input, Lambda, Add
from keras.models import Model
from keras import backend as K

class deep_q_learning_agent:
    def __init__(self, dueling):
        self.env = environment()
        
        # Fast pre-allocated NumPy Replay Buffer to speed up training loop
        self.states = np.zeros((memory_size, num_features), dtype=np.float32)
        self.actions = np.zeros(memory_size, dtype=np.int32)
        self.rewards = np.zeros(memory_size, dtype=np.float32)
        self.next_states = np.zeros((memory_size, num_features), dtype=np.float32)
        self.memory_counter = 0

        self.model = self.create_model(dueling)
        self.target_model = self.create_model(dueling)

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.loss_function = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_deepQ)

    def create_model(self, dueling):
        input_shape = (num_features,)
        X_input = Input(input_shape)  # input layer
        X = X_input

        X = Dense(512, input_shape=input_shape, activation="tanh")(X)  # first hidden layer

        X = Dense(256, activation="tanh")(X)  # second hidden layer

        X = Dense(64, activation="tanh")(X)  # third hidden layer

        if dueling:
            state_value = Dense(1, activation="linear")(X)
            state_value = Lambda(lambda s: tf.tile(s, [1, num_actions]))(state_value)

            action_advantage = Dense(num_actions, activation="linear")(X)
            action_advantage = Lambda(lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True))(action_advantage)

            X = Add()([state_value, action_advantage])
        else:
            X = Dense(num_actions, activation="linear")(X)  # output layer

        model = Model(inputs=X_input, outputs=X)  # create deep neural network
        return model

    def normalize_state(self, state):
        # Scale queue states from [0, 10] down to [0, 1] to prevent tanh saturation and speed up learning
        normalized = np.copy(state).astype(np.float32)
        if len(normalized.shape) == 2:
            normalized[:, 1] /= d_queue_size
            normalized[:, 2] /= e_queue_size
        else:
            normalized[1] /= d_queue_size
            normalized[2] /= e_queue_size
        return normalized

    def remember(self, state, action, reward, next_state):
        index = self.memory_counter % memory_size
        self.states[index] = np.reshape(state, (num_features,))
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = np.reshape(next_state, (num_features,))
        self.memory_counter += 1

    @tf.function
    def train_step(self, state_sample, action_sample, reward_sample, next_state_sample):
        # Double DQN action selection using online network
        next_q_online = self.model(next_state_sample, training=False)
        next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)
        
        # Double DQN value evaluation using target network
        next_q_target = self.target_model(next_state_sample, training=False)
        
        # Gather target Q-values corresponding to the online actions
        batch_indices = tf.range(tf.shape(next_actions)[0], dtype=tf.int32)
        indices = tf.stack([batch_indices, next_actions], axis=1)
        future_q_values = tf.gather_nd(next_q_target, indices)
        
        updated_q_values = reward_sample + gamma_deepQ * future_q_values

        mask = tf.one_hot(action_sample, num_actions)
        with tf.GradientTape() as tape:
            q_values = self.model(state_sample, training=True)
            q_action = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
            loss = self.loss_function(updated_q_values, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def replay(self):
        max_mem = min(self.memory_counter, memory_size)
        if max_mem < batch_size:
            return
        indices = np.random.choice(max_mem, size=batch_size, replace=False)
        
        state_sample = self.states[indices]
        action_sample = self.actions[indices]
        reward_sample = self.rewards[indices]
        next_state_sample = self.next_states[indices]

        self.train_step(
            tf.convert_to_tensor(state_sample, dtype=tf.float32),
            tf.convert_to_tensor(action_sample, dtype=tf.int32),
            tf.convert_to_tensor(reward_sample, dtype=tf.float32),
            tf.convert_to_tensor(next_state_sample, dtype=tf.float32)
        )

    def target_update(self):
        self.target_model.set_weights(self.model.get_weights())

    @tf.function
    def predict_q_values(self, state_tensor):
        return self.model(state_tensor, training=False)

    def get_action(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.get_possible_action())
        else:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            list_value = self.predict_q_values(state_tensor)[0].numpy()
            list_actions = self.env.get_possible_action()
            max_q = -float("inf")
            action = 0
            for action_t in list_actions:
                if list_value[action_t] >= max_q:
                    max_q = list_value[action_t]
                    action = action_t
            return action

    def learning(self):
        total_reward = 0
        for i in range(T):
            current_state = self.env.get_state_deep()
            current_state = np.reshape(current_state, (1, num_features))
            normalized_state = self.normalize_state(current_state)
            action = self.get_action(normalized_state)

            reward, next_state = self.env.perform_action_deep(action)
            next_state = np.reshape(next_state, (1, num_features))
            normalized_next_state = self.normalize_state(next_state)
            
            total_reward += reward
            self.remember(normalized_state, action, reward, normalized_next_state)

            self.replay()

            if (i+1) % update_target_network == 0:
                self.target_update()
            if (i+1) % step == 0:
                print("deep_q_learning Iteration " + str(i + 1) + " reward: " + str(total_reward / (i + 1)))

if __name__ == "__main__":
    agent = deep_q_learning_agent(dueling=True)
    agent.learning()