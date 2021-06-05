import logging
import numpy as np

import gym.spaces as spaces
from rlberry.agents import IncrementalAgent
from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from rlberry.utils.writers import PeriodicWriter


logger = logging.getLogger(__name__)


class UCBMQAgent(IncrementalAgent):
    """
    Implementation of the UCBMQ algorithm.

    Parameters
    ----------
    env : gym.Env
        Environment with discrete states and actions.
    n_episodes : int
        Number of episodes
    gamma : double, default: 1.0
        Discount factor in [0, 1].
    horizon : int
        Horizon of the objective function.
    bonus_scale_factor : double, default: 1.0
        Constant by which to multiply the exploration bonus, controls
        the level of exploration.
    bonus_type : {"simplified_bernstein"}
        Type of exploration bonus. Currently, only "simplified_bernstein"
        is implemented.
    """
    name = "UCBMQ"

    def __init__(self,
                 env,
                 n_episodes=1000,
                 horizon=100,
                 bonus_scale_factor=1.0,
                 bonus_type="simplified_bernstein",
                 debug=False,
                 **kwargs):
        # init base class
        IncrementalAgent.__init__(self, env, **kwargs)

        self.n_episodes = n_episodes
        self.horizon = horizon
        self.bonus_scale_factor = bonus_scale_factor
        self.bonus_type = bonus_type
        self.debug = debug

        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # maximum value
        r_range = self.env.reward_range[1] - self.env.reward_range[0]
        if r_range == np.inf or r_range == 0.0:
            logger.warning("{}: Reward range is  zero or infinity. ".format(self.name)
                           + "Setting it to 1.")
            r_range = 1.0

        self.v_max = np.zeros(self.horizon)
        self.v_max[-1] = r_range
        for hh in reversed(range(self.horizon-1)):
            self.v_max[hh] = r_range + self.v_max[hh+1]

        # initialize
        self.reset()

    def reset(self, **kwargs):
        H = self.horizon
        S = self.env.observation_space.n
        A = self.env.action_space.n

        # (s, a) visit counter
        self.N_sa = np.zeros((H, S, A))

        # Value functions
        self.V_bar = np.ones((H+1, S))
        self.V_bar[H, :] = 0
        self.Q = np.zeros((H, S, A))
        self.Q_bar = np.ones((H, S, A))
        self.V_hsa = np.ones((H, S, A, S))
        for hh in range(self.horizon):
            self.V_bar[hh, :] *= (self.horizon-hh)
            self.Q_bar[hh, :, :] *= (self.horizon-hh)
            self.V_hsa[hh, :, :, :] *= (self.horizon-hh)

        # ep counter
        self.episode = 0

        # useful object to compute total number of visited states & entropy of visited states
        self.counter = DiscreteCounter(self.env.observation_space,
                                       self.env.action_space)

        # info
        self._rewards = np.zeros(self.n_episodes)

        # default writer
        self.writer = PeriodicWriter(self.name,
                                     log_every=5*logger.getEffectiveLevel())

    def policy(self, state, hh=0, **kwargs):
        """ Recommended policy. """
        return self.Q_bar[hh, state, :].argmax()

    def _get_action(self, state, hh=0):
        """ Sampling policy. """
        return self.Q_bar[hh, state, :].argmax()

    def _compute_bonus(self, n, hh):
        if self.bonus_type == "simplified_bernstein":
            bonus = self.bonus_scale_factor * np.sqrt(1.0 / n) + self.v_max[hh] / n
            bonus = min(bonus, self.v_max[hh])
            return bonus
        else:
            raise ValueError(
                "Error: bonus type {} not implemented".format(self.bonus_type))

    def _update(self, state, action, next_state, reward, hh):
        H = self.horizon
        self.N_sa[hh, state, action] += 1
        nn = self.N_sa[hh, state, action]

        # learning rate
        alpha = 1.0/nn
        gamma = (H / (H+nn)) * ((nn-1)/nn)

        if self.debug:
            alpha = (self.horizon+1.0)/(self.horizon + nn)
            gamma = 0.0

        # bonus
        bonus = self._compute_bonus(nn, hh)

        # target and bias correction
        target = reward + self.V_bar[hh+1, next_state]
        bias_correction = self.V_bar[hh+1, next_state] - self.V_hsa[hh, state, action, next_state]

        # update Q
        self.Q[hh, state, action] = (1-alpha)*self.Q[hh, state, action] + alpha * target + gamma*bias_correction
        self.Q_bar[hh, state, action] = self.Q[hh, state, action] + bonus      # bonus here

        # Update V_bar  
        max_q = self.Q_bar[hh, state, :].max()
        prev_vbar = self.V_bar[hh, state]
        self.V_bar[hh, state] = max(min(max_q, prev_vbar), 0.0)

        if self.debug:
            self.V_bar[hh, state] = min(self.v_max[hh], self.Q_bar[hh, state, :].max())    

        # update V_hsa
        lr = alpha + gamma
        for ns in range(self.env.observation_space.n):
            self.V_hsa[hh, state, action, ns] = lr * self.V_bar[hh+1, ns] + (1-lr)*self.V_hsa[hh, state, action, ns]
            

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        for hh in range(self.horizon):
            action = self._get_action(state, hh)
            next_state, reward, done, _ = self.env.step(action)
            episode_rewards += reward  # used for logging only

            self.counter.update(state, action)

            self._update(state, action, next_state, reward, hh)

            state = next_state
            if done:
                break

        # update info
        ep = self.episode
        self._rewards[ep] = episode_rewards
        self.episode += 1

        # writer
        if self.writer is not None:
            self.writer.add_scalar("ep reward", episode_rewards, self.episode)
            self.writer.add_scalar("total reward", self._rewards[:ep].sum(), self.episode)
            self.writer.add_scalar("n_visited_states", self.counter.get_n_visited_states(), self.episode)

        # return sum of rewards collected in the episode
        return episode_rewards

    def partial_fit(self, fraction, **kwargs):
        assert 0.0 < fraction <= 1.0
        n_episodes_to_run = int(np.ceil(fraction*self.n_episodes))
        count = 0
        while count < n_episodes_to_run and self.episode < self.n_episodes:
            self._run_episode()
            count += 1

        info = {"n_episodes": self.episode,
                "episode_rewards": self._rewards[:self.episode]}

        return info
