#!/usr/bin/env python3
# TEAM MEMBERS:
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c
import argparse

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace):
    # Fix random seed
    np.random.seed(args.seed)
    #print(env.action_space)
    #print(env.observation_space)
    number_states = 4096
    number_actions = 2

    # :
    # - Create Q, a zero-filled NumPy array with shape [number of states, number of actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with the same shape,
    #   representing number of observed returns of a given (state, action) pair.
    Q = np.random.rand(number_states, number_actions)
    C = np.zeros((number_states, number_actions))

    for _ in range(args.episodes):
        # Perform episode, collecting states, actions and rewards
        states, actions, rewards = [], [], []
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # : Compute `action` using epsilon-greedy policy. Therefore,
            # with probability of `args.epsilon`, use a random action,
            # otherwise choose and action with maximum `Q[state, action]`.
            r = np.random.uniform()
            if args.epsilon > r:
                # take random action
                action = np.random.randint(0, number_actions)
            else:
                # take argmax action of current state
                action = np.argmax(Q[state])

            # Perform the action.
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # : Compute returns from the received rewards and update Q and C.
        G = 0
        # start at len() -2 because of starting at second last in algorithm and another -1 for indexing
        # go until -1 -> 0 is the last included one
        # go backwards with steps -1
        for t in range(len(rewards) - 1 - 1, -1, -1):
            G += rewards[t+1]
            C[states[t], actions[t]] += 1
            Q[states[t], actions[t]] += (1 / C[states[t], actions[t]]) * (G - Q[states[t], actions[t]])

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # : Choose a greedy action
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), args.seed)

    main(env, args)
