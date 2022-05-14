#!/usr/bin/env python3
# TEAM MEMBERS:
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c

import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=32, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")


class Agent:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # : Create a suitable model. The predict method assumes
        # it is stored as `self._model`.
        #
        # Using Adam optimizer with given `args.learning_rate` is a good default.
        model = tf.keras.Sequential()
        # shape of state: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
        model.add(tf.keras.layers.Input([4]))
        model.add(tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

        # no loss or optimizer, training is done manually in self.train
        model.compile()

        model.summary()
        self._model = model

    # Define a training method.
    #
    # Note that we need to use @tf.function for efficiency (using `train_on_batch`
    # on extremely small batches/networks has considerable overhead).
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # : Perform training, using the loss from the REINFORCE algorithm.
        # The easiest approach is to use the `sample_weight` argument of
        # tf.losses.Loss.__call__, but you can also construct the Loss object
        # with tf.losses.Reduction.NONE and perform the weighting manually.

        with tf.GradientTape() as tape:
            predictions = self._model(states)
            # compute loss with actions as "gold data" and predictions of actions with states as x
            loss = tf.losses.SparseCategoricalCrossentropy()(y_true=actions, y_pred=predictions, sample_weight=returns)

        variables = self._model.variables
        gradients = tape.gradient(loss, variables)

        for variable, gradient in zip(variables, gradients):
            variable.assign_sub(args.learning_rate * gradient)

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    print(args)

    # Construct the agent
    agent = Agent(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # : Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                state_expanded = np.expand_dims(state, axis=0)  # Introduction of batch dim necessary to pass it to input layer of self._model
                predictions = agent.predict(state_expanded)[0]  # index 0 because of added batch dim
                action = np.random.choice(2, p=predictions)

                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # : Compute returns from the received rewards
            returns = [sum(rewards[:i]) for i in range(len(rewards))]

            # : Add states, actions and returns to the training batch
            # important: extend to add all content and not append which appends the whole list as one element
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)

        # : Train using the generated batch.
        agent.train(batch_states, batch_actions, batch_returns)
    print(args)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # : Choose greedy action
            state_expanded = np.expand_dims(state, axis=0)  # Introduction of batch dim necessary to pass it to input layer of self._model
            predictions = agent.predict(state_expanded)[0]  # index 0 because of added batch dim
            action = np.argmax(predictions)
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed)

    main(env, args)
