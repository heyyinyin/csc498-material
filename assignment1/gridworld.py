import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(-1).reshape(x.shape[0], -1, 1)


class Gridworld():

    def __init__(self, state_transition, rewards):
        self.state_transition = state_transition
        self.rewards = rewards

        self.state = 0
        self.action_space = self.state_transition.shape[1]
        self.observation_space = self.state_transition.shape[0]

    def reset(self):
        self.state = 0
        return self.state

    def step(self, act):
        next_state_dist = self.state_transition[self.state, act]
        self.state = np.random.choice(range(self.observation_space), p=next_state_dist)
        rew = self.rewards[self.state]
        return self.state, rew, False, {}


class StochasticGridworld(Gridworld):
    def __init__(self):
        state_transition = np.array([
                [0, 1],
                [1, 3],
                [4, 5],
                [0, 3],
                [2, 7],
                [0, 1],
                [0, 1],
                [6, 7]]
            )
        prob_state_transition = np.ones((8, 2, 8), dtype=np.int) * 0.2/7
        for i, next_state in enumerate(state_transition):
            prob_state_transition[i, 0, next_state[0]] = 0.8
            prob_state_transition[i, 1, next_state[1]] = 0.8
        rewards = np.array([-5, 1, 2, 3, 4, 5, 6, 7])
        super().__init__(prob_state_transition, rewards)


class DeterministicGridworld(Gridworld):
    def __init__(self):
        state_transition = np.array([
                [0, 1],
                [1, 3],
                [4, 5],
                [0, 3],
                [2, 7],
                [2, 1],
                [0, 1],
                [6, 7]]
            )
        prob_state_transition = np.ones((8, 2, 8), dtype=np.int) * 0
        for i, next_state in enumerate(state_transition):
            prob_state_transition[i, 0, next_state[0]] = 1.
            prob_state_transition[i, 1, next_state[1]] = 1.
        rewards = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        super().__init__(prob_state_transition, rewards)


def make_random_gridworld(state_size, action_size):
    state_transitions = np.random.normal(scale =1.5, size=(state_size, action_size, state_size))
    state_transitions = softmax(state_transitions)
    rewards = np.random.normal(size=(state_size,))

    return Gridworld(state_transitions, rewards)
