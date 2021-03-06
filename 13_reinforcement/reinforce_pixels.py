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

import cart_pole_pixels_environment
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--episodes", default=2000, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--pooling", default="max", type=str, help="Type of pooling.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")


class Agent:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # : Create a suitable model. The predict method assumes
        # the policy network is stored as `self._model`.
        #
        # Apart from the model defined in `reinforce`, define also another
        # model for computing the baseline (with a single output without an activation).
        # (Alternatively, this baseline computation can be grouped together
        # with the policy computation in a single `tf.keras.Model`.)
        #
        # Using Adam optimizer with given `args.learning_rate` for both models
        # is a good default.

        if not args.recodex:

            # shared architecture
            inputs = tf.keras.layers.Input([80, 80, 3])
            # 40x40x8
            conv1 = self.bn_relu(tf.keras.layers.Conv2D(8, 3, 2, "same")(inputs))
            # 20x20x10
            conv2 = self.bn_relu(tf.keras.layers.Conv2D(10, 3, 2, "same")(conv1))
            # 10x10x15
            conv3 = self.bn_relu(tf.keras.layers.Conv2D(15, 3, 2, "same")(conv2))
            # 5x5x15
            pool_class = tf.keras.layers.MaxPooling2D if args.pooling == 'max' else tf.keras.layers.AveragePooling2D
            pooled = pool_class()(conv3)
            flatten = tf.keras.layers.Flatten()(pooled)
            hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(flatten)

            # create models with specific heads
            output_actions = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(hidden)
            self._policy_model = tf.keras.Model(inputs=inputs, outputs=output_actions)
            self._policy_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                loss=tf.losses.SparseCategoricalCrossentropy()
            )
            self._policy_model.summary()

            output_baseline = tf.keras.layers.Dense(1, activation=None)(hidden)
            output_baseline_reshaped = tf.reshape(output_baseline, [-1])
            self._baseline_model = tf.keras.Model(inputs=inputs, outputs=output_baseline_reshaped)
            self._baseline_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                loss=tf.losses.MeanSquaredError()
            )
            self._baseline_model.summary()
        else:
            self._policy_model = tf.keras.models.load_model("reinforce_pixels_policy.h5")
            self._baseline_model = tf.keras.models.load_model("reinforce_pixels_baseline.h5")

    @staticmethod
    def bn_relu(input):
        return tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(input))

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
        # : Perform training, using the loss from the REINFORCE with
        # baseline algorithm.
        # You should:
        # - compute the predicted baseline using the baseline model
        # - train the baseline model to predict `returns`
        # - train the policy model, using `returns - predicted_baseline` as
        #   the advantage estimate

        # A) TRAIN BASELINE MODEL
        with tf.GradientTape() as baseline_tape:
            baseline_predictions = self._baseline_model(states)
            baseline_loss = self._baseline_model.compiled_loss(y_true=returns, y_pred=baseline_predictions)
        self._baseline_model.optimizer.minimize(loss=baseline_loss, var_list=self._baseline_model.trainable_variables,
                                                tape=baseline_tape)

        # B) TRAIN POLICY MODEL
        with tf.GradientTape() as policy_tape:
            policy_predictions = self._policy_model(states)
            # compute loss with actions as "gold data" and predictions of actions with states as x
            policy_loss = self._policy_model.compiled_loss(y_true=actions, y_pred=policy_predictions,
                                                           sample_weight=returns - baseline_predictions)
        self._policy_model.optimizer.minimize(loss=policy_loss, var_list=self._policy_model.trainable_variables,
                                              tape=policy_tape)

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._policy_model(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the agent
    agent = Agent(env, args)

    if not args.recodex:
        # : Perform training
        for _ in range(args.episodes // args.batch_size):
            batch_states, batch_actions, batch_returns = [], [], []
            for _ in range(args.batch_size):
                # Perform episode
                states, actions, rewards = [], [], []
                state, done = env.reset(), False
                while not done:
                    if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                        env.render()

                    # (reinforce): Choose `action` according to probabilities
                    # distribution (see `np.random.choice`), which you
                    # can compute using `agent.predict` and current `state`.
                    state_expanded = np.expand_dims(state, axis=0)
                    predictions = agent.predict(state_expanded)[0]
                    action = np.random.choice(2, p=predictions)

                    next_state, reward, done, _ = env.step(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                    state = next_state

                # (reinforce): Compute returns from the received rewards
                returns = [sum(rewards[i:]) for i in range(len(rewards))]

                # (reinforce): Add states, actions and returns to the training batch
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_returns.extend(returns)

            # (reinforce): Train using the generated batch.
            agent.train(batch_states, batch_actions, batch_returns)

        agent._policy_model.save("reinforce_pixels_policy.h5")
        agent._baseline_model.save("reinforce_pixels_baseline.h5")
        print(args)

    # : Perform evaluation of a trained model.
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # : Choose an action
            state_expanded = np.expand_dims(state, axis=0)
            predictions = agent.predict(state_expanded)[0]
            action = np.argmax(predictions)
            state, reward, done, _ = env.step(action)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPolePixels-v0"), args.seed)

    main(env, args)
