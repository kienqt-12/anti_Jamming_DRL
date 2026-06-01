from environment import environment
import numpy as np
from parameters import *
import tensorflow as tf
from keras.layers import Dense, Input, Lambda, Add
from keras.models import Model
from keras import backend as K
from collections import deque

class deep_q_learning_agent:
    def __init__(self, dueling):
        self.env = environment()
        self.action_history = deque(maxlen=memory_size)
        self.state_history = deque(maxlen=memory_size)
        self.reward_history = deque(maxlen=memory_size)
        self.next_state_history = deque(maxlen=memory_size)

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

    def remember(self, state, action, reward, next_state):
        self.state_history.append(state)
        self.action_history.append(action)
        self.next_state_history.append(next_state)
        self.reward_history.append(reward)

    @tf.function
    def train_step(self, state_sample, action_sample, reward_sample, next_state_sample):
        future_rewards = self.target_model(next_state_sample, training=False)
        updated_q_values = reward_sample + gamma_deepQ * tf.reduce_max(future_rewards, axis=1)

        mask = tf.one_hot(action_sample, num_actions)
        with tf.GradientTape() as tape:
            q_values = self.model(state_sample, training=True)
            q_action = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
            loss = self.loss_function(updated_q_values, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def replay(self):
        if len(self.state_history) < batch_size:
            return
        indices = np.random.choice(range(len(self.state_history)), size=batch_size)
        state_sample = np.array([self.state_history[i] for i in indices]).reshape((batch_size, num_features))
        action_sample = np.array([self.action_history[i] for i in indices], dtype=np.int32)
        reward_sample = np.array([self.reward_history[i] for i in indices], dtype=np.float32)
        next_state_sample = np.array([self.next_state_history[i] for i in indices]).reshape((batch_size, num_features))

        self.train_step(
            tf.convert_to_tensor(state_sample, dtype=tf.float32),
            tf.convert_to_tensor(action_sample, dtype=tf.int32),
            tf.convert_to_tensor(reward_sample, dtype=tf.float32),
            tf.convert_to_tensor(next_state_sample, dtype=tf.float32)
        )

    def target_update(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.get_possible_action())
        else:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            list_value = self.model(state_tensor, training=False)[0].numpy()
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
            action = self.get_action(current_state)

            reward, next_state = self.env.perform_action_deep(action)
            next_state = np.reshape(next_state, (1, num_features))
            total_reward += reward
            self.remember(current_state, action, reward, next_state)

            self.replay()

            if (i+1) % update_target_network == 0:
                self.target_update()
            if (i+1) % step == 0:
                print("deep_q_learning Iteration " + str(i + 1) + " reward: " + str(total_reward / (i + 1)))

if __name__ == "__main__":
    agent = deep_q_learning_agent(dueling=True)
    agent.learning()